#PBS -A project_name
#PBS -k doe
#PBS -l walltime=07:00:00
#PBS -l filesystems=home:eagle
#PBS -l select=4
#PBS -q by-gpu
module use /soft/modulefiles; module load conda; conda activate base
cd BlockFusion
python triplane_VAE_training.py --expname="various_loss_combination_exp" \
 --description="L1 + L2 + Geo loss combination (No MS-SSIM) (lr scheduling and KL-annealing roughly following the record from logs/triplane_model_g/20250703-184718)\
 Multiple Sets of sample coords for calculating geometry loss" \
 --batch_size=24 --epochs=2100 --ckpt_freq=500 --model_config=model_g \
 --mae_loss_weight 1.0 --mse_loss_weight 0.5 --ms_ssim_loss_weight 0.0 --lpips_loss_weight 0.0 \
 --kl_loss_weight_values 0.000001 0.00001 0.00005 --kl_loss_weight_epochs 0 500 1000 \
 --geometry_loss_weight 0.8 --scheduler_type MultiStepLR \
 --init_lr 0.0001 --lr_gamma 0.5 --milestones 500 1000 1213 1500 1650 1750 1850 1925 2000 2050