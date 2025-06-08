import os
import torch
import torch.nn.functional as F
import numpy as np

import logging
import argparse
from datetime import datetime

from autoencoder_2D_origin import VAE
from timevarying_data_helper import LatentWeightDataset
from fit_triplane.visualize_triplane import plot_single_channel

# redefinition of traning pipeline for multiple input volumes
def train_vae(vae_model, train_dataloader, optimizer, scheduler, init_lr, lr_decay, lr_gamma, epochs=100, tensorboard_writer=None, console_logger=None, run_dir=None, ckpt_freq=100, resume_epoch=0):

    vae_model.train()
    
    # Add gradient scaling for mixed precision training
    # But loss would become NaN, so disable it for now
    scalar = torch.amp.GradScaler("cuda", enabled=False)
    min, max = train_dataloader.dataset.get_value_range()
    value_range = max - min
    
    for epoch in range(epochs):
        epoch = epoch + resume_epoch
        
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        running_loss = 0.0
        total_elems = 0
        
        # mini-batch or SGD (with small batch as one sample) training
        # since we do optimization after each batch
        for batch_idx, raw_data in enumerate(train_dataloader):
            
            # Add gradient scaling for mixed precision training
            # But loss would become NaN, so disable it for now
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                output = vae_model(raw_data)
                # for debugging
                # if epoch == 5 and batch_idx == 0:
                #     import pdb; pdb.set_trace()
                #     plot_single_channel(output[0][0][0][0], f"epoch{epoch}_batch{batch_idx}", save_path=f"epoch{epoch}_batch{batch_idx}")
                # reconstructed results is the first element of the output (output[0])
                recon_loss = F.mse_loss(output[0], raw_data)
                kl_loss = vae_model.loss_function(*output)
                loss = recon_loss + kl_loss
                # loss = recon_loss
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            running_recon_loss += recon_loss.item() * raw_data.shape[0]
            running_kl_loss += kl_loss.item() * raw_data.shape[0]
            running_loss += running_recon_loss + running_kl_loss
            total_elems += raw_data.shape[0]
            
            # TODO: need to sperate PSNR evaluation from each volume (cause currently has four volumes in one batch)
            # console_logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, KL loss: {kl_loss.item():0,.6f}, Reconstruction PSNR: {(20 * torch.log10(raw_data.max() - raw_data.min() / torch.sqrt(recon_loss))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
            if console_logger is not None:
                console_logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, KL loss: {kl_loss.item():0,.6f}, Reconstruction PSNR: {(20 * np.log10(value_range / np.sqrt(recon_loss.item()))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
            else:
                print(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, KL loss: {kl_loss.item():0,.6f}, Reconstruction PSNR: {(20 * np.log10(value_range / np.sqrt(recon_loss.item()))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
            # import pdb; pdb.set_trace()
        
        # calculate the average loss for the epoch
        assert total_elems == len(train_dataloader.dataset), "total_elems should be equal to the dataset size"
        last_recon_loss = running_recon_loss / total_elems
        last_kl_loss = running_kl_loss / total_elems
        last_loss = running_loss / total_elems
        last_PSNR = 20 * np.log10(value_range / np.sqrt(last_recon_loss))
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("Loss/Train_Recon", last_recon_loss, epoch)
            tensorboard_writer.add_scalar("Loss/Train_KL", last_kl_loss, epoch)
            tensorboard_writer.add_scalar("Loss/Train", last_loss, epoch)
            tensorboard_writer.add_scalar("Loss/Train_PSNR", last_PSNR, epoch)
            tensorboard_writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)
        
        # adjust learning rate
        scheduler.step()
        
        # save the model at checkpoint
        if (epoch % ckpt_freq == (ckpt_freq - 1)) or (epoch == (epochs + resume_epoch) - 1):
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'init_lr': init_lr,
                'lr_decay': lr_decay,
                'lr_gamma': lr_gamma,
                'loss': last_loss,
                'recon_loss': last_recon_loss,
                'kl_loss': last_kl_loss,
                'PSNR': last_PSNR,
            }, os.path.join(run_dir, f"vae_model_epoch_{epoch}.ckpt"))

            # # save one of the reconstructed volume data     
            # with open(os.path.join(run_dir, f"reconstructed_volume_epoch_{epoch}.data"), "wb") as f:
            #     # write the reconstructed volume data to file
            #     # only try to store the last batch's first volume
            #     output[0][0].clamp(raw_data.min(), raw_data.max()).detach().cpu().numpy().astype(np.float32).tofile(f)
        

if __name__ == "__main__":
    vae_config = {"kl_std": 0.25,
                  "kl_weight": 0.001,
                  "plane_shape": [3, 32, 128, 128],
                  "z_shape": [4, 32, 32],
                  "num_heads": 16,
                  "transform_depth": 1}
    vae_model = VAE(vae_config).cuda()
    
    n_timesteps = 90
    batch_size = 8
    init_lr = 0.0001
    lr_decay = 300
    lr_gamma = 0.5
    epoch = 1500
    ckpt_freq = 500
    expname = "triplane_VAE_ch_32"
    
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

    # load pretrained tri-plane here
    loaded_model = torch.load("fit_triplane/ch_32_saved_model.ckpt")
    triplane_weights = [loaded_model['triplane_state_dict'][f"{idx}.triplane"] for idx in range(n_timesteps)]
    triplane_weights = torch.cat(triplane_weights, dim=0)
    # TODO: check whether it makes sense to pass vae_config["plane_shape"] into LatentWeightDataset
    # since it is originally designed to receive latent weights in 3D space (hash grid)
    # Need to check how does the original VAE receive input:
    # [batch size, 3 (triplanes), #channels, res, res] or [batch size, #channels, res, res * 3 (triplanes)]
    # seems both work(?)
    train_dataloader = torch.utils.data.DataLoader(
        LatentWeightDataset(
        triplane_weights,
        vae_config["plane_shape"]),
        batch_size=batch_size,
        shuffle=True)
    # input_tensor = torch.randn(2, 3, 32, 128, 128).cuda()
    input_tensor = next(iter(train_dataloader)).cuda()
    out = vae_model(input_tensor)
    loss = vae_model.loss_function(*out)
    print("loss: {}".format(loss))
    print("z shape: {}".format(out[-1].shape))
    print("reconstruct shape: {}".format(out[0].shape))
    samples = vae_model.sample(2)
    print("samples shape: {}".format(samples[0].shape))
    
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=lr_gamma)
    
    train_vae(vae_model, train_dataloader, optimizer, scheduler, init_lr, lr_decay, lr_gamma, epoch, tensorboard_writer, console_logger, run_dir, ckpt_freq, 0)