import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat, einsum
from einops.layers.torch  import Rearrange, Reduce

def zero_module(module:nn.Module) ->  nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def identity_module(module:nn.Module) -> nn.Module:
    nn.init.dirac_(module.weight)
    return module

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super()._init_()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.LayerNorm(dim),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        channel_avg = torch.mean(x, dim=(2,3))
        return x * self.gate(channel_avg)[:, :, None, None]

class NACBlock(nn.Sequential):
    """Normalization, Activation, Convolution"""
    def __init__(self, in_chs:int, out_chs:int, act_fn, norm_groups=32):
        super().__init__(nn.GroupNorm(num_groups=min(norm_groups, in_chs), num_channels=in_chs), act_fn,
                         nn.Conv2d(in_chs, out_chs, 3, 1, 1))

class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embedding (for including time information)"""
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        super().__init__()

        half_dim = time_emb_dims // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

        ts  = torch.arange(total_time_steps, dtype=torch.float32)

        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=True),
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
        )

    def forward(self, time):
        return self.time_blocks(time)

class OldAttentionBlock(nn.Module):
    def __init__(self, channels=64, norm_groups=32, n_heads=4, dropout=0.0):
        super().__init__()
        self.channels = channels

        self.group_norm = nn.GroupNorm(num_groups=min(norm_groups,channels), num_channels=channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=n_heads, batch_first=True, dropout=dropout)

    def forward(self, x, *args, **kwargs):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = rearrange(h, "b c h w -> b (h w) c")
        # h = h.reshape(B, self.channels, H * W).swapaxes(1, 2)  # [B, C, H, W] --> [B, C, H * W] --> [B, H*W, C]
        h, _ = self.mhsa(h, h, h)  # [B, H*W, C]
        # h = h.swapaxes(2, 1).view(B, self.channels, H, W)  # [B, C, H*W] --> [B, C, H, W]
        h = rearrange(h, "b (h w) c -> b c h w ")
        return x + h

class CrossAttention(nn.Module):
    def __init__(self, channels=64, n_heads=4, condition_dim=4, dropout=0.0, **kwargs):
        super().__init__()
        self.channels = channels
        self.condition_dim = condition_dim

        self.norm = nn.LayerNorm(channels)
        self.proj_cond = nn.Linear(condition_dim, channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=n_heads, batch_first=True, dropout=dropout)

    def forward(self, x, *args, context=None):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).swapaxes(1, 2)  # [B, C, H, W] --> [B, C, H * W] --> [B, H*W, C]
        # context = context or x
        context = self.proj_cond(context)

        q, k, v = self.norm(x), self.norm(context), self.norm(context)
        att, _ = self.mhsa(q, k, v)  # [B, H*W, C]
        x = (x + att).swapaxes(2, 1).reshape(B, C, H, W)  # [B, H*W, C] --> [B, C, H, W]
        return x

def manual_attn(q,k,v):
    scale = 1 / math.sqrt(math.sqrt(q.shape[1]))
    weight = einsum(q*scale, k*scale,
        "b c t,b c s->b t s", 
    )*scale  # More stable with f16 than dividing afterwards     # (b*heads) x M x M
    weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
    A = einsum(weight, v, "b t s,b c s->b c t")  # (b*heads) x ch x length
    return A

class AttentionBlock(nn.Module):
    def __init__(self, channels=64, n_heads=4, norm_groups=32, dropout=0.0, **kwargs):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(norm_groups, channels), num_channels=channels)
        self.qkv_conv = nn.Conv1d(channels, channels*3, 1)
        self.to_heads = Rearrange("b d (n_heads c_t) -> (b n_heads) d c_t", n_heads=n_heads)
        self.from_heads = Rearrange("(b n_heads) d c_t -> b d (n_heads c_t)", n_heads=n_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
        self.dropout_p = dropout
    
    def forward(self, x, *args, **kwargs):
        xnorm = self.norm(x)
        b, c, h, w = xnorm.shape
        x = rearrange(x, "b c h w -> b c (h w)")
        xnorm = rearrange(xnorm, "b c h w -> b c (h w)")
        qkv = self.qkv_conv(xnorm)
        q, k, v = qkv.chunk(3, dim=1) # each is now [B, C, H*W]
        q, k, v = self.to_heads(q), self.to_heads(k), self.to_heads(v) # [B*heads, H*W, C/heads]
        attn_mask = torch.ones((q.shape[1], q.shape[1]), device=xnorm.device, dtype=torch.bool)
        A = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p, attn_mask=attn_mask)
        A = self.proj_out(self.from_heads(A))
        x = A + x
        return rearrange(x, "b c (h w) -> b c h w", h=h)

class ResBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, dropout=0.0, act_fn=None, norm_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.out_channels:
            self.match_input = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        else:
            self.match_input = nn.Identity()
        act_fn = act_fn or nn.SiLU()
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=min(norm_groups, self.in_channels), num_channels=self.in_channels),
            act_fn,
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.GroupNorm(num_groups=min(norm_groups,out_channels), num_channels=out_channels),
            nn.Dropout2d(p=dropout),
            act_fn,
            zero_module(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same")),
        )
    
    def forward(self, x):
        h = self.layers(x)
        h = h + self.match_input(x)
        return h

class ResidualAndOut(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, y):
        return self.fn(x+y)

class ScaleShift(nn.Module):
    def __init__(self, fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)
    def forward(self, x, y):
        scale, shift = torch.chunk(y, 2, dim=1)
        x = self.fns[0](x) * (1 + scale) + shift
        x = self.fns[1](x)
        return x

class ResBlockWithTime(nn.Module):
    def __init__(self, in_chs, out_chs, dropout_rate=0.0, time_emb_dims=512,
                  use_scale_shift_norm=False, norm_groups=32):
        super().__init__()

        self.act_fn = nn.SiLU()
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=norm_groups, num_channels=in_chs)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_1 = nn.Conv2d(in_chs, out_chs,
                                kernel_size=3, stride=1, padding=1)

        # Group 2 time embedding
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=out_chs)

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=norm_groups, num_channels=out_chs)
        self.conv_2 = zero_module(nn.Conv2d(out_chs, out_chs,
                                kernel_size=3, stride=1, padding="same"))
        
        if use_scale_shift_norm:
            self.add_emb = ScaleShift([self.normlize_2, self.conv_2])
        else:
            self.add_emb = ResidualAndOut(nn.Sequential(self.normlize_2, self.conv_2))

        if in_chs != out_chs:
            self.match_input = nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1)
        else:
            self.match_input = nn.Identity()

    def forward(self, x, t, **kwargs):
        # group 1
        h = self.act_fn(self.normlize_1(x))
        h = self.dropout(h)
        h = self.conv_1(h)

        # timestep embedding
        emb_out = self.dense_1(self.act_fn(t))[:, :, None, None]

        h = self.add_emb(h, emb_out)
        
        # Residual
        return h + self.match_input(x)

class DownSample(nn.Module):
    def __init__(self, channels:int, use_conv=False):
        super().__init__()
        if use_conv:
            self.downsample = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = nn.AvgPool2d(2, 2)
    def forward(self, x, *args, **kwargs):
        return self.downsample(x)

class UpSample(nn.Module):
    """Up sample by 2, then convolve"""
    def __init__(self, in_channels:int):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            identity_module(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)))
    def forward(self, x, *args, **kwargs):
        return self.upsample(x)

class UpSamplePixelShuffle(nn.Module):
    """Up sample by 2 using pixel shuffle layer"""
    def __init__(self, in_channels:int):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*4,
                      kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, stride=1, padding=1),
            )
    def forward(self, x, *args, **kwargs):
        return self.upsample(x)

def pairwise_distance(a:torch.Tensor, b:torch.Tensor):
    # basically (a-b)^2 = a^2 -2ab + b^2
    return torch.sum(a**2, dim=1, keepdim=True) - 2*(torch.matmul(a, b.t())) + torch.sum(b**2, dim=1)

class VectorQuantizer(nn.Module):
    def __init__(self, n_vectors:int, latent_dim:int):
        super().__init__()
        self.n_vectors = n_vectors
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(self.n_vectors, self.latent_dim)
        # why uniform ? idk
        self.embedding.weight.data.uniform_(-1.0 / self.n_vectors, 1.0 / self.n_vectors)

    def forward(self, z_in):
        z = z_in.permute(0, 2, 3, 1).contiguous() # [B, C, H, W] -> [B, H, W, C]
        # C must be divisible by latent_dim
        z_flattened = z.view(-1, self.latent_dim) # [B, H, W, C] -> [B*H*W, latent_dim]
        
        distances = pairwise_distance(z_flattened, self.embedding.weight) # [B*H*W, n_vectors]
        closest_indices = torch.argmin(distances, dim=1) # [B*H*W, n_vectors] -> [B*H*W]
        quantized = self.embedding(closest_indices).view(z.shape) # [B*H*W, latent_dim] -> [B, H, W, C]
        quantized = quantized.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]

        z_q = z_in + (quantized - z_in).detach() # Equivalent to "copy gradient"

        return z_q, quantized