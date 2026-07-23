#PBS -A <project name>
#PBS -k doe
#PBS -l walltime=15:30:00
#PBS -l filesystems=home:eagle
#PBS -l select=8
#PBS -q by-gpu
module use /soft/modulefiles; module load conda; conda activate blockfusion
cd kctung/Projects/BlockFusion
MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --rdzv_id=my_job --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 \
 triplane_VAE_training_shadow.py --expname="VAE_vorts1_baseline" \
 --description="vorts1 baseline for CAD/Graphics paper" \
 --batch_size=6 --epochs=800 --ckpt_freq=200 \
 --init_lr 0.00005 --lr_gamma 0.5 --patience 50 \
 --model_config=model_g --scheduler_type ReduceLROnPlateau \
 --mae_loss_weight 1.0 --mse_loss_weight 0.0 --ms_ssim_loss_weight 0.0 --lpips_loss_weight 0.0 \
 --geometry_loss_weight 0.0 --geometry_loss_sample_size 0 \
 --kl_loss_weight_values 0.000001 0.00001 0.00005 --kl_loss_weight_epochs 0 250 500 \
 --pretrained_triplane_file_path logs/vorts1_triplane_overfitting/20260721-090419/pure_triplane_model_permuted.pt