import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, Conv3DAwareUNet
import numpy as np
from timevarying_data_helper import LatentWeightDataset
import os

import logging
from datetime import datetime

def train_diffusion(model, train_dataloader, noise_scheduler, optimizer, scheduler, init_lr, lr_decay, lr_gamma, epochs, tensorboard_writer=None, console_logger=None, run_dir=None, ckpt_freq=100, resume_epoch=0):
    # Training loop
    model.train()
    
    # min, max = train_dataloader.dataset.get_value_range()
    # value_range = max - min
    
    num_batches = len(train_dataloader)
    
    for epoch in range(epochs):
        epoch = epoch + resume_epoch

        for batch_idx, clean_images in enumerate(train_dataloader):
            # need to reshape (stack 3 triplanes) along the last dimension -> [B, C, H, W * 3 (triplanes)]
            clean_images = clean_images.cuda()
            clean_images = clean_images.permute(0, 2, 3, 1, 4).reshape(clean_images.shape[0], clean_images.shape[2], clean_images.shape[3], clean_images.shape[4]*3)
            # print("check clean images shape:")
            # import pdb; pdb.set_trace()
            # Sample noise and timesteps
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],)).long().cuda()

            # Add noise to images
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Predict the noise
            noise_pred = model(noisy_images, timesteps).sample

            # Loss = predicted noise vs true noise
            loss = F.mse_loss(noise_pred, noise)

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if step % 100 == 0:
            # print(f"Epoch {epoch} | Step {batch_idx} | Loss: {loss.item():.4f}")
            steps = epoch * num_batches + batch_idx
            loss_val = loss.item()
            # PSNR_val = 20 * np.log10(value_range / np.sqrt(loss_val))
            
            if console_logger is not None:
                console_logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Step: {steps}, Total loss: {loss_val:0,.6f}, LR: {scheduler.get_last_lr()[0]}")
                # console_logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Step: {steps}, Total loss: {loss_val:0,.6f}, Reconstruction PSNR: {PSNR_val:0,.4f}, LR: {scheduler.get_last_lr()[0]}")
            else:
                print(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss_val:0,.6f}, LR: {scheduler.get_last_lr()[0]}")
                # print(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss_val:0,.6f}, Reconstruction PSNR: {PSNR_val:0,.4f}, LR: {scheduler.get_last_lr()[0]}")
            
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("Loss/Train", loss_val, steps)
                # tensorboard_writer.add_scalar("Loss/Train_PSNR", PSNR_val, steps)
                tensorboard_writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], steps)
        
        # adjust learning rate
        scheduler.step()
        
        # save the model at checkpoint
        if (epoch % ckpt_freq == (ckpt_freq - 1)) or (epoch == (epochs + resume_epoch) - 1):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'init_lr': init_lr,
                'lr_decay': lr_decay,
                'lr_gamma': lr_gamma,
                'loss': loss_val,
                # 'PSNR': PSNR_val,
            }, os.path.join(run_dir, f"Diffusion_UNet_model_epoch_{epoch}.ckpt"))
        

if __name__ == "__main__":

    # Configs
    # image_size = 32
    # TODO: use argparse to receive the arguments
    n_timesteps = 90
    batch_size = 5
    epoch = 200
    init_lr = 1e-4
    lr_decay = 100
    lr_gamma = 0.5
    ckpt_freq = 100
    expname = "UNet_diffusion_exp"
    
    # create directory for saving logs
    base_dir = "./logs"
    os.makedirs(base_dir, exist_ok=True)
    expname_dir = os.path.join(base_dir, expname)
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

    # Dataset and DataLoader
    # load pretrained tri-plane here
    loaded_model = torch.load("fit_triplane/ch_64_saved_model.ckpt")
    triplane_weights = [loaded_model['triplane_state_dict'][f"{idx}.triplane"] for idx in range(n_timesteps)]
    triplane_weights = torch.cat(triplane_weights, dim=0)
    # TODO: plane_shape should be provided with loaded_model (?)
    plane_shape = [3, 64, 128, 128]
    train_dataloader = torch.utils.data.DataLoader(
        LatentWeightDataset(
        triplane_weights,
        plane_shape),
        batch_size=batch_size,
        shuffle=True)
    # input_tensor = torch.randn(2, 3, 32, 128, 128).cuda()
    input_tensor = next(iter(train_dataloader)).cuda()
    
    model = Conv3DAwareUNet(
        sample_size=plane_shape[2:],
        in_channels=plane_shape[1],
        out_channels=plane_shape[1],
        block_out_channels=(128, 256, 384, 512)
        # rest of the arguments uses default values
    ).cuda()
    input_tensor = input_tensor.permute(0, 2, 3, 1, 4).reshape(batch_size, plane_shape[1], plane_shape[2], plane_shape[3]*3)
    output = model(input_tensor, 100)
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

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=lr_gamma)
    
    train_diffusion(model, train_dataloader, noise_scheduler, optimizer, scheduler, init_lr, lr_decay, lr_gamma, epoch, tensorboard_writer, console_logger, run_dir, ckpt_freq, 0)