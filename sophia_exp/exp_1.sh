#PBS -A insitu
#PBS -k doe
#PBS -l walltime=03:30:00
#PBS -l filesystems=home:eagle
#PBS -l select=1
#PBS -q by-gpu
module use /soft/modulefiles; module load conda; conda activate base
cd BlockFusion
python triplane_VAE_training.py --expname=test --description="test the correctness of refactored triplane_VAE_training.py" \
 --batch_size=4 --epochs=1500 --ckpt_freq=1500 --init_lr=0.0001 --lr_decay=250 --lr_gamma=250 --model_config=model_a