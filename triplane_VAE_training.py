import os
import torch
import torch.nn.functional as F
import numpy as np

import logging
import argparse
from datetime import datetime
import importlib

from autoencoder_2D_origin import VAE_no_KL
from timevarying_data_helper import LatentWeightDataset
from fit_triplane.visualize_triplane import plot_single_channel

check_plane_idx = 50
vis_triplane_freq = 50

def parse_args():
    parser = argparse.ArgumentParser(description="Train a autoencoder on triplanes")
    parser.add_argument("--expname", type=str, default="triplane_revised_autoencoder_training_test", help="Experiment name")
    parser.add_argument("--description", type=str, default="Test Model_a with layernorm after spatial transformer", help="Description to experiment")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train")
    parser.add_argument("--ckpt_freq", type=int, default=1000, help="Checkpoint frequency")
    parser.add_argument("--init_lr", type=float, default=1e-3, help="Initial learning rate for training")
    # parser.add_argument("--lr_decay", type=int, default=3000, help="Learning rate decay frequency")
    parser.add_argument("--lr_gamma", type=float, default=0.2, help="Learning rate decay factor")
    parser.add_argument("--patience", type=int, default=150, help="Number of patience epochs to decay learning rate")
    parser.add_argument("--resume_training_dir", type=str, default=None, help="Directory to resume training from")
    parser.add_argument("--resume_model_file_name", type=str, default=None, help="Model file name to resume training from")
    parser.add_argument("--model_config", type=str, default="model_a", help="Model config file name")
    return parser.parse_args()

# redefinition of traning pipeline for multiple input volumes
def train_vae(vae_model, train_dataloader, optimizer, scheduler, epochs=100, tensorboard_writer=None, console_logger=None, run_dir=None, ckpt_freq=100, resume_epoch=0):

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
                output = vae_model(raw_data[0])
                # output[0] = output[0].clamp(min, max)
                # for debugging
                if epoch % vis_triplane_freq == (vis_triplane_freq - 1) or epoch < 5:
                    channel = 16
                    # for training on all volumes
                    if check_plane_idx in raw_data[1]:
                        for plane in range (3):
                            plot_single_channel(output[0][torch.where(raw_data[1] == check_plane_idx)[0].item()][plane][channel].detach(), f"epoch{epoch}_plane{plane}", save_path=os.path.join(run_dir, f"epoch{epoch}_plane{plane}.png"))
                    # for only one volume
                    # for plane in range (3):
                    #     plot_single_channel(output[0][0][plane][channel].detach(), f"epoch{epoch}_plane{plane}", save_path=os.path.join(run_dir, f"epoch{epoch}_plane{plane}.png"))
                    
                # reconstructed results is the first element of the output (output[0])
                recon_loss = F.mse_loss(output[0], raw_data[0])
                kl_loss = vae_model.module.loss_function(*output)
                loss = recon_loss + kl_loss
                # loss = recon_loss
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            running_recon_loss += recon_loss.item() * raw_data[0].shape[0]
            running_kl_loss += kl_loss.item() * raw_data[0].shape[0]
            running_loss += running_recon_loss + running_kl_loss
            total_elems += raw_data[0].shape[0]
            
            # TODO: need to sperate PSNR evaluation from each volume (cause currently has four volumes in one batch)
            # console_logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, KL loss: {kl_loss.item():0,.6f}, Reconstruction PSNR: {(20 * torch.log10(raw_data.max() - raw_data.min() / torch.sqrt(recon_loss))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
            if console_logger is not None:
                console_logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, KL loss: {kl_loss.item():0,.6f}, Reconstruction PSNR: {(20 * np.log10(value_range / np.sqrt(recon_loss.item()))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
                # console_logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, Reconstruction PSNR: {(20 * np.log10(value_range / np.sqrt(recon_loss.item()))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
            else:
                print(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, KL loss: {kl_loss.item():0,.6f}, Reconstruction PSNR: {(20 * np.log10(value_range / np.sqrt(recon_loss.item()))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
                # print(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, Reconstruction PSNR: {(20 * np.log10(value_range / np.sqrt(recon_loss.item()))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
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
        scheduler.step(last_loss)
        
        # save the model at checkpoint
        if (epoch % ckpt_freq == (ckpt_freq - 1)) or (epoch == (epochs + resume_epoch) - 1):
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
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
    args = parse_args()
    
    cfg = importlib.import_module(f"autoencoder_config.triplane.{args.model_config}")
    
    vae_model = torch.nn.DataParallel(VAE_no_KL(cfg.vae_config, cfg.encoder_dims, cfg.feature_size_encoder, cfg.decoder_dims,
                                                cfg.feature_size_decoder, cfg.fpn_encoders_layer_dim_idx, cfg.fpn_decoders_layer_dim_idx,
                                                cfg.fpn_encoders_down_idx, cfg.fpn_encoders_up_idx, cfg.fpn_decoders_down_idx,
                                                cfg.fpn_decoders_up_idx, cfg.block_config)).cuda()
    
    n_timesteps = 90
    # batch_size = 1
    
    # resume training from ckpt
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
    
    # prepare python logger
    logging.basicConfig(filename=os.path.join(run_dir, "console_log.log"),
                    format='%(asctime)s %(message)s',
                    filemode=logging_file_md)
    console_logger = logging.getLogger()
    console_logger.setLevel(logging.DEBUG)

    # to suppress matplotlib logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # load pretrained tri-plane here
    loaded_model = torch.load("fit_triplane/ch_32_saved_model.ckpt")
    triplane_weights = [loaded_model['triplane_state_dict'][f"{idx}.triplane"] for idx in range(n_timesteps)]
    triplane_weights = torch.cat(triplane_weights, dim=0)
    
    # normalize triplane value to -1, 1
    # normalization for training only one volume
    # triplane_weights[50:51][0] = (triplane_weights[50:51][0] - triplane_weights[50:51][0].min()) / (triplane_weights[50:51][0].max() - triplane_weights[50:51][0].min())
    # triplane_weights[50:51][0] = triplane_weights[50:51][0] * 2 - 1
    
    # normalization for training all volumes
    triplane_weights = (triplane_weights - triplane_weights.min()) / (triplane_weights.max() - triplane_weights.min())
    triplane_weights = triplane_weights * 2 - 1
    
    train_dataloader = torch.utils.data.DataLoader(
        LatentWeightDataset(
        # test training for only one volume
        # triplane_weights[50:51],
        triplane_weights,
        cfg.vae_config["plane_shape"]),
        batch_size=args.batch_size,
        shuffle=True)
    # input_tensor = torch.randn(2, 3, 32, 128, 128).cuda()
    input_tensor = next(iter(train_dataloader))[0].cuda()
    out = vae_model(input_tensor)
    loss = vae_model.module.loss_function(*out)
    print("loss: {}".format(loss))
    print("z shape: {}".format(out[-1].shape))
    print("reconstruct shape: {}".format(out[0].shape))
    samples = vae_model.module.sample(2)
    print("samples shape: {}".format(samples[0].shape))
    
    model_arch_str = str(vae_model)
    
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=args.init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=args.lr_gamma, patience=args.patience)
    
    # resume training
    if args.resume_training_dir and args.resume_model_file_name:
        resume_epoch = loaded_ckpt["epoch"] + 1
        console_logger.debug(f"Resume training from {args.resume_training_dir}/{args.resume_model_file_name} at Epoch {resume_epoch}")
        console_logger.debug(f"Batch Size: {args.batch_size}, Epochs: {args.epochs}, Checkpoint Frequency: {args.ckpt_freq}")
        vae_model.load_state_dict(loaded_ckpt["model_state_dict"])
        optimizer.load_state_dict(loaded_ckpt["optimizer_state_dict"])
        # check whether users pass lr for resume training
        # TODO: this doesn't work if passing argument like this in command line: --init_lr=0.0001
        # only works like this: --init_lr 0.0001
        import sys
        if "--init_lr" in sys.argv and "--lr_gamma" in sys.argv and "--patience" in sys.argv:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.init_lr
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=args.lr_gamma, patience=args.patience)
            console_logger.debug(f"Reset learning rate to {args.init_lr}, patience epochs to {args.patience}, learning rate decay factor to {args.lr_gamma}")
        elif "--init_lr" in sys.argv or "--lr_gamma" in sys.argv or "--patience" in sys.argv:
            raise RuntimeError("Only reset either init_lr, lr_gamma, or patience for resuming training, please reset three arguments at the same time")
        else:
            scheduler.load_state_dict(loaded_ckpt["scheduler_state_dict"])
            console_logger.debug(f"Use original learning rate and scheduler settings")
        
    # training from scratch
    else:
        resume_epoch = 0
        tensorboard_writer.add_text("Model/Architecture", f"```\n{model_arch_str}\n```", global_step=0)
    
        console_logger.debug("Experiment description: " + args.description)
        console_logger.debug(f"Batch Size: {args.batch_size}, Epochs: {args.epochs}, Checkpoint Frequency: {args.ckpt_freq}")
        console_logger.debug(f"Initial Learning rate: {args.init_lr}, Patience Epochs: {args.patience}, Learning rate Decay Factor: {args.lr_gamma}")
        console_logger.debug(f"Model config: autoencoder_config.triplane.{args.model_config}")
    
    
    train_vae(vae_model, train_dataloader, optimizer, scheduler, args.epochs, tensorboard_writer, console_logger, run_dir, args.ckpt_freq, resume_epoch)