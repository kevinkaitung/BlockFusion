import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, Conv3DAwareUNet, Conv3DAwareUNet2DConditionModel
import numpy as np
from timevarying_data_helper import LatentWeightDataset, ShadowVolumesMetaDataset, ShadowLightingDirectionsDataset
import os
import math

import logging
from datetime import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Diffusion on time-varying data")
    parser.add_argument("--expname", type=str, default="UNet_diffusion_exp", help="Experiment name")
    parser.add_argument("--description", type=str, default="Initial test on diffusion model training", help="Description to experiment")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train")
    parser.add_argument("--ckpt_freq", type=int, default=1000, help="Checkpoint frequency")
    parser.add_argument("--init_lr", type=float, default=None, help="Initial learning rate for training")
    # parser.add_argument("--lr_decay", type=int, default=3000, help="Learning rate decay frequency")
    parser.add_argument("--lr_gamma", type=float, default=None, help="Learning rate decay factor")
    parser.add_argument("--patience", type=int, default=None, help="Number of patience epochs to decay learning rate")
    parser.add_argument("--resume_training_dir", type=str, default=None, help="Directory to resume training from")
    parser.add_argument("--resume_model_file_name", type=str, default=None, help="Model file name to resume training from")
    parser.add_argument("--latent_triplanes_file_path", type=str, default=None, help="File Path to latent triplanes")
    # load pretrained triplane just to get the lighting direction of each instance
    # TODO: incorporate the lighting direction info into latent triplanes file
    parser.add_argument("--pretrained_triplane_file_path", type=str, default=None, help="File Path to Pretrained Triplanes Model")
    return parser.parse_args()

# simple embedding for testing
class NumberEmbedder(torch.nn.Module):
    def __init__(self, embed_dim_in, embed_dim_out):
        super().__init__()
        self.proj = torch.nn.Linear(embed_dim_in, embed_dim_out)

    def forward(self, numbers):
        return self.proj(numbers)

num_freqs = 64
# number of freqs * 3 coordinates * 2 (sin and cos)
embed_dim = num_freqs * 3 * 2

# copy from diffusers/models/embeddings.py
class FourierEmbedder(torch.nn.Module):
    def __init__(self, num_freqs=64, temperature=100):
        super().__init__()

        self.num_freqs = num_freqs
        self.temperature = temperature

        freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)
        freq_bands = freq_bands[None, None, None]
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def __call__(self, x):
        x = self.freq_bands * x.unsqueeze(-1)
        return torch.stack((x.sin(), x.cos()), dim=-1).permute(0, 1, 3, 4, 2).reshape(*x.shape[:2], -1)

def train_diffusion(model, train_dataloader, shadow_meta_dataset, noise_scheduler, optimizer, scheduler, epochs, positional_embedder, tensorboard_writer=None, console_logger=None, run_dir=None, ckpt_freq=100, resume_epoch=0):
    # Training loop
    model.train()
    
    # min, max = train_dataloader.dataset.get_value_range()
    # value_range = max - min
    
    num_batches = len(train_dataloader)
    
    for epoch in range(epochs):
        epoch = epoch + resume_epoch

        running_loss = 0.0
        total_elems = 0

        for batch_idx, clean_images in enumerate(train_dataloader):
            indices = clean_images[1]
            clean_images = clean_images[0]
            # only permute for raw triplanes and reshape (stack 3 triplanes) along the last dimension -> [B, C, H, W * 3 (triplanes)]
            # clean_images = clean_images.permute(0, 2, 3, 1, 4).reshape(clean_images.shape[0], clean_images.shape[2], clean_images.shape[3], clean_images.shape[4]*3)
            # print("check clean images shape:")
            # import pdb; pdb.set_trace()
            # Sample noise and timesteps
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],)).long().cuda()

            # Add noise to images
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            pos_embed = positional_embedder(shadow_meta_dataset[indices])
            # make the shape to be [batch_size, sequence_length (currently one for representing one light), feature_dim]
            pos_embed = pos_embed[0].unsqueeze(1)
            # Predict the noise
            noise_pred = model(noisy_images, timesteps, pos_embed).sample

            # Loss = predicted noise vs true noise
            loss = F.mse_loss(noise_pred, noise)

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item() * clean_images.shape[0]
            total_elems += clean_images.shape[0]
            
            steps = epoch * num_batches + batch_idx
            
            if console_logger is not None:
                console_logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Step: {steps}, Total loss: {loss.item():0,.6f}, LR: {scheduler.optimizer.param_groups[0]['lr']}")
            else:
                print(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, LR: {scheduler.optimizer.param_groups[0]['lr']}")
        
        # calculate the average loss for the epoch
        assert total_elems == len(train_dataloader.dataset), "total_elems should be equal to the dataset size"
        last_loss = running_loss / total_elems
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("Loss/Train", last_loss, epoch)
            tensorboard_writer.add_scalar("Learning Rate", scheduler.optimizer.param_groups[0]['lr'], epoch)

        # adjust learning rate
        scheduler.step(last_loss)
        
        # save the model at checkpoint
        if (epoch % ckpt_freq == (ckpt_freq - 1)) or (epoch == (epochs + resume_epoch) - 1):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': last_loss,
            }, os.path.join(run_dir, f"Diffusion_UNet_model_epoch_{epoch}.ckpt"))
        

if __name__ == "__main__":
    args = parse_args()
    
    # n_timesteps = 90
    
    if args.resume_training_dir and args.resume_model_file_name:
        run_dir = args.resume_training_dir
        logging_file_md = 'a'
        loaded_ckpt = torch.load(os.path.join(args.resume_training_dir, args.resume_model_file_name))
    elif (args.resume_training_dir and not args.resume_model_file_name) or (not args.resume_training_dir and args.resume_model_file_name):
        RuntimeError("Missing resume training directory or model file name to resume training")
    else:
        # create directory for saving logs
        base_dir = "./logs"
        os.makedirs(base_dir, exist_ok=True)
        expname_dir = os.path.join(base_dir, args.expname)
        os.makedirs(expname_dir, exist_ok=True)
        run_dir = os.path.join(expname_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        logging_file_md = 'w'
    
    # create tensorboard logger
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_writer = SummaryWriter(log_dir=run_dir)
    
    # # prepare python logger
    logging.basicConfig(filename=os.path.join(run_dir, "console_log.log"),
                    format='%(asctime)s %(message)s',
                    filemode=logging_file_md)
    console_logger = logging.getLogger()
    console_logger.setLevel(logging.DEBUG)

    # to suppress matplotlib logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # Dataset and DataLoader
    # load pretrained tri-plane here (for raw triplane training)
    # loaded_model = torch.load("fit_triplane/ch_32_saved_model.ckpt")
    # triplane_weights = [loaded_model['triplane_state_dict'][f"{idx}.triplane"] for idx in range(n_timesteps)]
    # triplane_weights = torch.cat(triplane_weights, dim=0)
    
    loaded_model = torch.load(args.latent_triplanes_file_path)
    latent_weights = loaded_model["weights_latent_space"]
    # normalize latent triplanes to -1~1 for diffusion training
    latent_weights = (latent_weights - latent_weights.min()) / (latent_weights.max() - latent_weights.min())
    latent_weights = latent_weights * 2 - 1
    latent_weights = latent_weights.cuda()
    
    # TODO: think about where to put z_shape. with Unet arch def or with latent_triplanes.pt?
    # z_shape = loaded_model["z_shape"] # sample shape: [4, 32, 32]
    z_shape = [4, 32, 32]
    dataset = LatentWeightDataset(
        latent_weights,
        [z_shape[0], z_shape[1], z_shape[2] * 3])
    # shadow_meta_dataset = ShadowVolumesMetaDataset(
    #     raw_data_dir=path_to_dataset_directory,
    #     raw_data_filename_prefix="shadow",
    #     file_ext="json",
    #     n_instances=len(dataset),
    # )
    shadow_meta_dataset = ShadowLightingDirectionsDataset(
        lighting_dirs=torch.load(args.pretrained_triplane_file_path)["light_dir_cartesian"]
    )
    # number_embedder = NumberEmbedder(24, embed_dim).cuda()
    positional_embedder = FourierEmbedder(num_freqs=num_freqs).cuda()
    train_dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size,
        shuffle=True)
    
    # example for raw triplanes
    # plane_shape = [3, 32, 128, 128]
    # train_dataloader = torch.utils.data.DataLoader(
    #     LatentWeightDataset(
    #     triplane_weights,
    #     plane_shape),
    #     batch_size=batch_size,
    #     shuffle=True)
    # input_tensor = torch.randn(2, 3, 32, 128, 128)
    
    # model = Conv3DAwareUNet(
    #     sample_size=z_shape[1:],
    #     in_channels=z_shape[0],
    #     out_channels=z_shape[0],
    #     block_out_channels=(128, 256, 512, 1024),
    #     layers_per_block=3
    #     # rest of the arguments uses default values
    # ).cuda()
    
    model = Conv3DAwareUNet2DConditionModel(
        sample_size=z_shape[1:],
        in_channels=z_shape[0],
        out_channels=z_shape[0],
        down_block_types=("DownBlock2D", "SimpleCrossAttnDownBlock2D", "SimpleCrossAttnDownBlock2D", "SimpleCrossAttnDownBlock2D"),
        up_block_types=("SimpleCrossAttnUpBlock2D", "SimpleCrossAttnUpBlock2D", "SimpleCrossAttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(128, 256, 512, 1024),
        layers_per_block=2,
        cross_attention_dim=embed_dim,
        # rest of the arguments uses default values
    ).cuda()
    
    input_tensor = next(iter(train_dataloader))
    
    pos_embed = positional_embedder(shadow_meta_dataset[input_tensor[1]])
    # make the shape to be [batch_size, sequence_length (currently one for representing one light), feature_dim]
    pos_embed = pos_embed[0].unsqueeze(1)
    # only permute for raw triplanes to comply with the model input shape
    # input_tensor = input_tensor.permute(0, 2, 3, 1, 4).reshape(batch_size, plane_shape[1], plane_shape[2], plane_shape[3]*3)
    output = model(input_tensor[0], 100, pos_embed)
    print(f"example output: {output.sample.shape}")
    
    # # UNet model
    # model = UNet2DModel(
    #     sample_size=image_size,
    #     in_channels=3,
    #     out_channels=3,
    #     layers_per_block=2,
    #     block_out_channels=(64, 128, 128),
    #     down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    #     up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D")
    # ).to(device)

    # Diffusion noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=257)

    if args.resume_training_dir and args.resume_model_file_name:
        resume_epoch = loaded_ckpt["epoch"] + 1
        console_logger.debug(f"Resume training from {args.resume_training_dir}/{args.resume_model_file_name} at Epoch {resume_epoch}")
        console_logger.debug(f"Batch Size: {args.batch_size}, Epochs: {args.epochs}, Checkpoint Frequency: {args.ckpt_freq}")
        model.load_state_dict(loaded_ckpt["model_state_dict"])
        
        # placeholder to create optimizer and scheduler instances
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=150)
        optimizer.load_state_dict(loaded_ckpt["optimizer_state_dict"])
        # check whether users pass lr for resume training
        if args.init_lr and args.lr_gamma and args.patience:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.init_lr
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=args.lr_gamma, patience=args.patience)
            console_logger.debug(f"Reset learning rate to {args.init_lr}, patience epochs to {args.patience}, learning rate decay factor to {args.lr_gamma}")
        elif args.init_lr or args.lr_gamma or args.patience:
            raise RuntimeError("Only reset either init_lr, lr_gamma, or patience for resuming training, please reset three arguments at the same time")
        else:
            scheduler.load_state_dict(loaded_ckpt["scheduler_state_dict"])
            console_logger.debug(f"Use original learning rate and scheduler settings")
    
    # training from scratch
    else:
        resume_epoch = 0
        
        if args.init_lr and args.lr_gamma and args.patience:
            # Optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=args.lr_gamma, patience=args.patience)
        else:
            raise RuntimeError("Please specify init_lr, lr_gamma, and patience for training from scratch.")
        
        console_logger.debug("Experiment description: " + args.description)
        console_logger.debug(f"Batch Size: {args.batch_size}, Epochs: {args.epochs}, Checkpoint Frequency: {args.ckpt_freq}")
        console_logger.debug(f"Initial Learning rate: {args.init_lr}, Patience Epochs: {args.patience}, Learning rate Decay Factor: {args.lr_gamma}")
    
    train_diffusion(model, train_dataloader, shadow_meta_dataset, noise_scheduler, optimizer, scheduler, args.epochs, positional_embedder, tensorboard_writer, console_logger, run_dir, args.ckpt_freq, resume_epoch)