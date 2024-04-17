import gc, math, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from torchmetrics.multimodal import CLIPImageQualityAssessment
from diffusers import UNet2DConditionModel, DDPMScheduler, get_cosine_schedule_with_warmup, AutoencoderKL, VQModel
from data import SimpleImageDataModule, MultiImageDataModule
from layers import ResBlock, DownSample, AttentionBlock, NACBlock, Rearrange
from models import UNetConditionalCrossAttn, UNetConditionalCat
from sampler import ResShiftDiffusion

class DiffuserSRResShift(L.LightningModule):
    def __init__(self,
                 base_channels=128,
                 base_channels_multiples=(1,2,2,4),
                 apply_attention = (True, True, True, True),
                 n_layers=1, dropout_rate=0.0, scale_factor=4,
                 cross_attention_dim=768,
                lr=1e-4, scheduler_type="one_cycle", warmup_steps=500,
                vae_chkp="madebyollin/sdxl-vae-fp16-fix",
                use_scale_shift_norm=False,
                n_heads=4,
                vae_quantization=False,
                pixel_shuffle=False,
                use_cross_attn=False,
                timesteps=15, resshift_p=0.3, kappa=2.0):
        super().__init__()
        self.save_hyperparameters()
        self.scale_factor = scale_factor
        self.sampler = ResShiftDiffusion(timesteps, resshift_p, kappa)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.scheduler_type = scheduler_type
        self.vae_quantization = vae_quantization
        if vae_quantization:
            self.vae = VQModel.from_pretrained(vae_chkp)
            in_chs = 3
        else:
            self.vae = AutoencoderKL.from_pretrained(vae_chkp)
            in_chs = 4
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad_(False)
        
        cond_channels=3
        self.unet = UNetConditionalCat(timesteps, input_channels=in_chs, cond_channels=cond_channels, output_channels=in_chs,
                                        base_channels=base_channels,
                                        num_res_blocks=n_layers,
                                        n_heads=n_heads,
                                        base_channels_multiples=base_channels_multiples,
                                        apply_attention=apply_attention, dropout_rate=dropout_rate,
                                        pixel_shuffle=pixel_shuffle)
        self.cond_encoder = nn.Identity()

        self.metrics = {
            'psnr' : PeakSignalNoiseRatio(data_range=(-1,1)),
            'lpips' : LearnedPerceptualImagePatchSimilarity(net_type="vgg"),
            "ssim":  StructuralSimilarityIndexMeasure(data_range=(-1,1)),
        }

    def training_step(self, batch, batch_idx):
        if not isinstance(batch, (tuple, list)):
            high_res = batch
            low_res = F.interpolate(high_res + 0.1*torch.randn_like(high_res),
                                    scale_factor=1./self.scale_factor, mode='bicubic', antialias=True)
            cond = F.interpolate(low_res, scale_factor=self.scale_factor//4, mode='bicubic', antialias=True)
        elif len(batch) == 1:
            high_res = batch[0]
            low_res = F.interpolate(high_res + 0.1*torch.randn_like(high_res),
                                    scale_factor=1./self.scale_factor, mode='bicubic', antialias=True)
            cond = F.interpolate(low_res, scale_factor=self.scale_factor//4, mode='bicubic', antialias=True)
        elif len(batch) == 2:
            low_res, high_res = batch
            cond = F.interpolate(low_res, scale_factor=self.scale_factor//4, mode='bicubic', antialias=True)
        else:
            low_res, visible, high_res = batch
            cond = F.interpolate(visible, scale_factor=1.0/4, mode='bicubic', antialias=True)
        up_low_res = F.interpolate(low_res, scale_factor=self.scale_factor, mode='bicubic', antialias=True)
            
        with torch.no_grad():
            encoded_high_res = self.vae.encode(high_res, return_dict=False)[0]
            encoded_low_res = self.vae.encode(up_low_res, return_dict=False)[0]
            if not self.vae_quantization:
                encoded_high_res = encoded_high_res.mode()
                encoded_low_res = encoded_low_res.mode()

        t = torch.randint(low=1, high=self.sampler.timesteps, size=(high_res.shape[0],), device=high_res.device)
        xnoise = self.sampler.add_noise(encoded_high_res, encoded_low_res, torch.randn_like(encoded_high_res), t)

        x0_pred = self.unet(xnoise, t, self.cond_encoder(cond))
        loss = F.mse_loss(encoded_high_res, x0_pred)
        self.log("train_loss", loss)

        if getattr(self, "log_batch_cond", None) is None and self.trainer.is_global_zero:
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_image("high_res", 
                                                make_grid(high_res*0.5+0.5, nrow=4),self.global_step)
                self.logger.experiment.add_image("bicubic", 
                                                make_grid(up_low_res.clamp(-1,1)*0.5+0.5, nrow=4),self.global_step)
            
            self.log_batch_high_res = high_res.clone().detach()
            self.log_batch_high_res.requires_grad_(False)
            self.log_batch_cond = cond.clone().detach()
            self.log_batch_cond.requires_grad_(False)
            self.log_batch_encoded_low_res = encoded_low_res.clone().detach()
            self.log_batch_encoded_low_res.requires_grad_(False)

        if (self.trainer.is_global_zero and
            (self.global_step % self.trainer.log_every_n_steps) == 0):
            self.do_log(encoded_high_res.shape, self.log_batch_high_res.device)
        return loss

    @rank_zero_only
    @torch.inference_mode()
    def do_log(self, size, device):
        xnoise = self.sampler.prior_sample(self.log_batch_encoded_low_res,
                                           torch.randn_like(self.log_batch_encoded_low_res))

        self.unet.eval(); self.cond_encoder.eval()
        pred = self.predict(xnoise, self.log_batch_cond)
        self.unet.train(); self.cond_encoder.train()
        
        pred = self.vae.decode(pred, return_dict=False)[0].clamp(-1,1)
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image("pred",
                                            make_grid((pred*0.5+0.5).cpu(), nrow=4),
                                            self.global_step)
        gc.collect()
        torch.cuda.empty_cache()

        self.metrics['psnr'].to(device)
        psnr = self.metrics['psnr'](pred, self.log_batch_high_res)
        self.metrics['ssim'].to(device)
        ssim = self.metrics['ssim'](pred, self.log_batch_high_res)
        self.metrics['lpips'].to(device)
        lpips = self.metrics['lpips'](pred, self.log_batch_high_res)
        self.log_dict({'train_psnr':psnr,  'train_ssim': ssim, 'train_lpips':lpips}, rank_zero_only=True)

    @torch.inference_mode()
    def predict(self, xnoise:torch.Tensor, cond:torch.Tensor):
        cond_encoded = self.cond_encoder(cond)
        
        for time_step in reversed(range(self.sampler.timesteps)):
            ts = torch.ones(xnoise.shape[0], dtype=torch.long, device=xnoise.device) * time_step
            x0_pred = self.unet(xnoise, ts, cond_encoded)
            xnoise = self.sampler.backward_step(xnoise, x0_pred, ts)
        return xnoise
        
    def configure_optimizers(self):
        params = list(self.cond_encoder.parameters()) + list(self.unet.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        if self.scheduler_type is None:
            return [optimizer]
        if self.scheduler_type == "cosine_warmup":
            scheduler = get_cosine_schedule_with_warmup(optimizer, self.warmup_steps,
                                                        num_training_steps=self.trainer.estimated_stepping_batches)
        if self.scheduler_type == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                                            total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        with torch.inference_mode():
            if not isinstance(batch, (tuple, list)):
                high_res = batch
                low_res = F.interpolate(high_res + 0.1*torch.randn_like(high_res),
                                        scale_factor=1./self.scale_factor, mode='bicubic', antialias=True)
                cond = F.interpolate(low_res, scale_factor=self.scale_factor//4, mode='bicubic', antialias=True)
            elif len(batch) == 1:
                high_res = batch[0]
                low_res = F.interpolate(high_res + 0.1*torch.randn_like(high_res),
                                        scale_factor=1./self.scale_factor, mode='bicubic', antialias=True)
                cond = F.interpolate(low_res, scale_factor=self.scale_factor//4, mode='bicubic', antialias=True)
            elif len(batch) == 2:
                low_res, high_res = batch
                cond = F.interpolate(low_res, scale_factor=self.scale_factor//4, mode='bicubic', antialias=True)
            else:
                low_res, visible, high_res = batch
                cond = F.interpolate(visible, scale_factor=1.0/4, mode='bicubic', antialias=True)
            up_low_res = F.interpolate(low_res, scale_factor=self.scale_factor, mode='bicubic', antialias=True)

            encoded_low_res = self.vae.encode(up_low_res, return_dict=False)[0]
            if not self.vae_quantization:
                encoded_low_res = encoded_low_res.mode()

            xnoise = self.sampler.prior_sample(encoded_low_res, torch.randn_like(encoded_low_res))

            pred = self.predict(xnoise, cond)

            pred = self.vae.decode(pred, return_dict=False)[0].clamp(-1, 1)
            
            self.metrics['psnr'].to(pred.device)
            psnr = self.metrics['psnr'](pred, high_res)
            self.metrics['ssim'].to(pred.device)
            ssim = self.metrics['ssim'](pred, high_res)
            self.metrics['lpips'].to(pred.device)
            lpips = self.metrics['lpips'](pred, high_res)
            self.log_dict({'valid_psnr':psnr, 'hp_metric':psnr,
                  'valid_ssim':ssim, 'valid_lpips':lpips}, sync_dist=True, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx):
        with torch.inference_mode():
            if not isinstance(batch, (tuple, list)):
                low_res = batch
                cond = F.interpolate(low_res, scale_factor=self.scale_factor//4, mode='bicubic', antialias=True)
            elif len(batch) == 1:
                low_res = batch[0]
                cond = F.interpolate(low_res, scale_factor=self.scale_factor//4, mode='bicubic', antialias=True)
            else:
                low_res, visible = batch
                cond = F.interpolate(visible, scale_factor=1.0/4.0, mode='bicubic', antialias=True)
            up_low_res = F.interpolate(low_res, scale_factor=self.scale_factor, mode='bicubic', antialias=True)
            encoded_low_res = self.vae.encode(up_low_res, return_dict=False)[0]
            if not self.vae_quantization:
                encoded_low_res = encoded_low_res.mode()
            xnoise = self.sampler.prior_sample(encoded_low_res, torch.randn_like(encoded_low_res))
            pred = self.predict(xnoise, cond)
            pred = self.vae.decode(pred, return_dict=False)[0].clamp(-1, 1)
            return pred
    
def main():
    checkpoint_callback = ModelCheckpoint(save_top_k=-1, every_n_epochs=1, monitor="hp_metric")
    trainer_defaults = dict(enable_checkpointing=True, callbacks=[checkpoint_callback],
                            enable_progress_bar=False, log_every_n_steps=5_000)
    
    cli = LightningCLI(model_class=DiffuserSRResShift,
                       trainer_defaults=trainer_defaults)
if __name__ == "__main__":
    main()