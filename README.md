# BlockFusion for Volumetric Scalar field data

### Environment Setup (if running on Sophia):
Use this module:
```
module use /soft/modulefiles; module load conda; conda activate base
```
And install missing packages:
```
pip install --user easydict
```

### Stage 1 Training - Triplane Overfitting:

Please go into `./fit_triplane` and follow the instructions.

### Stage 2 Training - Variational Autoencoder Training on Triplanes

To train Variational Autoencoder, run the following sample command (Please specify `--pretrained_triplane_file_path`):
```
python triplane_VAE_training.py --expname="various_loss_combination_exp" \
 --description="L1 + L2 + Geo loss combination (No MS-SSIM) (lr scheduling and KL-annealing roughly following the record from logs/triplane_model_g/20250703-184718)" \
 --batch_size=6 --epochs=2100 --ckpt_freq=500 --model_config=model_g \
 --mae_loss_weight 1.0 --mse_loss_weight 0.5 --ms_ssim_loss_weight 0.0 --lpips_loss_weight 0.0 \
 --kl_loss_weight_values 0.000001 0.00001 0.00005 --kl_loss_weight_epochs 0 500 1000 \
 --geometry_loss_weight 0.8 --scheduler_type MultiStepLR \
 --init_lr 0.0001 --lr_gamma 0.5 --milestones 500 1000 1213 1500 1650 1750 1850 1925 2000 2050 \
 --pretrained_triplane_file_path path_to_pretrained_triplanes_file
```

Notes: VAE ckpt models can be large in size (about 9 GB), make sure to have enough space to save all ckpt files.

To inference and reconstruct triplanes with pre-trained Variational Autoencoder, run the following sample command (Please specify `--expdir` and `--model_file_name`):
```
python triplane_VAE_inference.py --expdir path_to_the_experiment_directory_created_when_training_VAE --model_file_name filename_of_the_model_used_to_inference --model_config model_g
```

For distributed training and jobs submission example, you can refer to `./sophia_exp/exp_mae_mse_geo_example.sh` (Please specify project name at the first line of the bash script)

# BlockFusion: Expandable 3D Scene Generation using Latent Tri-plane Extrapolation 

### ACM Transactions on Graphics (SIGGRAPH'24)

### [Project page](https://yang-l1.github.io/blockfusion/) | [Paper](https://arxiv.org/abs/2401.17053) | [Video](https://www.youtube.com/watch?v=PxIBtd6G0mA) 

This repo contains the official implementation for the paper "BlockFusion: Expandable 3D Scene Generation using Latent Tri-plane Extrapolation".

<img src="img/inference.gif" alt="drawing" width="350"/>
<img src="img/teaser.gif" alt="drawing" width="500"/>

### Installation
[Pytorch3d](https://github.com/facebookresearch/pytorch3d) is required for postprocess. Come to offical page for installation instructions.

We modified [diffusers](https://github.com/huggingface/diffusers) to adapt triplane structure. Users should build based on the ```./src```
```
# clone this repo 
python setup.py install
```


### Pretrained model
Pretrained weights of MLP, VAE, and Diffuser(Condition/Unconditioned). [Download the model here [10GB]](https://drive.google.com/file/d/19cuQihXzxq9fcM_drMWMhmwHBdT3B8fA/view?usp=sharing)
and extract to ```./checkpoints```.


### Inference


To do uncondtional inference, run 
```
python unconditioned_prediction.py --batch 4 --save output/uncond
```


To do single block conditional inference, run 

```
python conditioned_prediction.py --layout samplelayouts/exp1_32-56-24/0_0.npy --save output/cond
```

We provide some sample layouts for demo. If you would like to draw your own layout maps, you can use ```drawtkinter.py``` to create your own layout. Here is a tutorial:

![visualization](img/tutorial.gif)

The unit of the graduations are in meters, furnitures need to be in reasonable scale. Walls should surround the floor and furniture should be placed on the floor.

The output layouts directory named after ```expname_xscale-zscale-stride```, xscale-zscale-stride are given in decimeter.

Usage:
```
python drawtkinter.py --h 56 --w 56 --expname draw
```

Full pipeline(conditional prediction + extrapolation + resample) is contained in ```fullpipeline.py```. Before running, specifying the layout directory.


```
python fullpipeline.py --layout samplelayouts/exp2_32-56-24 --resample 15 --save output
```


## Citation

If you find our code or paper helps, please consider citing:
```
@article{blockfusion,
  title={BlockFusion: Expandable 3D Scene Generation using Latent Tri-plane Extrapolation},
  author={Wu, Zhennan and Li, Yang and Yan, Han and Shang, Taizhang and Sun, Weixuan and Wang, Senbo and Cui, Ruikai and Liu, Weizhe and Sato, Hiroyuki and Li, Hongdong and Ji, Pan},
  journal={ACM Transactions on Graphics},
  volume={43},
  number={4},
  year={2024},
  doi={10.1145/3658188}
 }      
```
            
