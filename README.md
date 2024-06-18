# Thermal Super-Resolution using Diffusion Models

This repository contains the implementation and pre-trained weights for the paper *Exploring the usage of diffusion models for thermal image super-resolution: a generic, uncertainty-aware approach for guided and non-guided schemes*.

## Inference
First download the checkpoints from the [releases](https://github.com/alcros33/ThermalSuperResolution/releases/tag/weights) tab
Place the LR images in a folder, lets call it `LR` (in the case of guided super-resolution, also place all the visible image in another folder, lets call it `visible`, be sure that matching images have the same names).
```
python inference.py checkpoint.pth --bs 8 -i LR visible -o rough_results
```
Then, since this a two-model approach use the refiner model to further enhance the results
```
python inference.py refiner_checkpoint.pth --refiner --bs 8 -i LR visible rough_results -o results
```

## Training
If you want to train the model in another dataset you may start by finetuning the one pretrained in imagenet (available in the [releases](https://github.com/alcros33/ThermalSuperResolution/releases/tag/weights) tab).
Multiple configuration files are provided in the [config](config) folder, adjust accordingly and run using
```
python diffusion_resshift.py fit --config my_config.yaml
```
If you instead want to train/finetune the refiner model, you may need to generate rough_predictions from the diffusion model using the code described in the inference section and adjust its configuration file accordingly.
```
python unet_refiner.py fit --config my_refiner_config.yaml
```

## License

Distributed under the MIT License. See `LICENSE` for more information.
