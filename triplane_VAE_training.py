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
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from fit_triplane.fit import Triplane, Network
from easydict import EasyDict as edict
import json
from timevarying_data_helper import SampleTimevaryingDataset

check_plane_idx = 50
vis_triplane_freq = 50
regenerate_sampled_points_freq = 10

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
    parser.add_argument("--patience", type=int, default=150, help="Number of patience epochs to decay learning rate (For ReduceLROnPlateau)")
    parser.add_argument("--milestones", type=int, nargs='*', default=None, help="List of epoch indices to decay learning rate (For MultiStepLR)")
    parser.add_argument("--resume_training_dir", type=str, default=None, help="Directory to resume training from")
    parser.add_argument("--resume_model_file_name", type=str, default=None, help="Model file name to resume training from")
    parser.add_argument("--model_config", type=str, default="model_a", help="Model config file name")
    parser.add_argument("--scheduler_type", type=str, default=None, help="Scheduler types: MultiStepLR or ReduceLROnPlateau")
    
    parser.add_argument("--mae_loss_weight", type=float, default=None, help="Weights of mae_loss") # default = 1.0
    parser.add_argument("--mse_loss_weight", type=float, default=None, help="Weights of mse_loss") # default = 1.0
    parser.add_argument("--ms_ssim_loss_weight", type=float, default=None, help="Weights of ms_ssim_loss") # default = 0.1
    parser.add_argument("--lpips_loss_weight", type=float, default=None, help="Weights of lpips_loss") # default = 1.0
    parser.add_argument("--geometry_loss_weight", type=float, default=None, help="Weights of geometry_loss") # default = 0.35
    parser.add_argument("--kl_loss_weight_values", type=float, nargs='*', default=None, help="Weight values of kl_loss") # default = [0.00001]
    parser.add_argument("--kl_loss_weight_epochs", type=int, nargs='*', default=None, help="Epochs to change weights of kl_loss") # default = [0]

    parser.add_argument("--pretrained_triplane_file_path", type=str, default="fit_triplane/ch_32_saved_model.ckpt", help="File Path to Pretrained Triplanes Model")
    return parser.parse_args()

def regenerate_sampled_points(dataset_for_sampling):
    # TODO: think about should I use different set of coordinates for different timestep volumes
    # currently only use one set of coords for all timesteps volumes
    sample_coords = [torch.rand([dataset_for_sampling.sample_batch_size, 3], dtype=torch.float32).cuda() for i in range(len(dataset_for_sampling))]
    target_values = [dataset_for_sampling.sample(i, sample_coords[i]) for i in range(len(dataset_for_sampling))]
    
    return sample_coords, target_values

def calculate_geometry_loss(net, triplane, timestep_indices, sample_coords, target_values, debug=False):
    losses = []
    for i in range(len(timestep_indices)):
        outputs = net(triplane[i](sample_coords[timestep_indices[i]], 0))
        losses.append(F.l1_loss(outputs, target_values[timestep_indices[i]].view(outputs.shape)))
    if debug:
        torch.save({"sample_coords":sample_coords[timestep_indices[i]],
                    "timestep_indices":timestep_indices[i],
                    "target_values":target_values[timestep_indices[i]],
                    "outputs":outputs,
                    "net":net,
                    "triplane":triplane},
                    "debug_tensor.pt")
    return torch.mean(torch.stack(losses), dim=0)

def create_triplane_model(config, batch_size):
    with open(config, 'r') as f:
        config = json.load(f)
    config = edict(config)
    # assert len(config.fixmlp) > 0

    net = Network(
        d_in=config.channel,
        d_hid=config.n_hid,
        n_layers=config.n_layers,
        d_out=config.n_labels,
        init_type="geo_init",
    ).cuda()

    # instantiate multiple triplanes (each timestep has its own triplane)
    # create batch_size triplanes (to align with VAE training batch_size)
    triplane = [Triplane(
        reso=config.resolution,
        channel=config.channel,
        init_type="geo_init",
        objname=None,
    ).cuda() for _ in range(batch_size)]
    triplane = torch.nn.ModuleList(triplane)

    return net, triplane

def load_params_to_triplane(triplane_models, triplane_params, original_min, original_max, normalized_min, normalized_max):
    # normalize VAE-recon triplane weights from the value range of -1~1 back to their original value range
    triplane_params = ((triplane_params - normalized_min) / (normalized_max - normalized_min)) * (original_max - original_min) + original_min
    for i in range(len(triplane_params)):
        with torch.no_grad():
            triplane_models[i].triplane.copy_(triplane_params[i:i+1])

# redefinition of traning pipeline for multiple input volumes
def train_vae(vae_model, train_dataloader, optimizer, scheduler, epochs=100, 
              tensorboard_writer=None, console_logger=None, run_dir=None, ckpt_freq=100, resume_epoch=0,
              mae_loss_weight=1.0, mse_loss_weight=1.0, ms_ssim_loss_weight=0.1, lpips_loss_weight=1.0, kl_loss_weight_values=[0.00001], kl_loss_weight_epochs=[0],
              geometry_loss_weight=0.35, dataset_for_sampling=None, net=None, triplane=None, original_triplane_min=0.0, original_triplane_max=1.0):

    vae_model.train()
    
    # Add gradient scaling for mixed precision training
    # But loss would become NaN, so disable it for now
    scalar = torch.amp.GradScaler("cuda", enabled=False)
    min, max = train_dataloader.dataset.get_value_range()
    value_range = max - min
    
    current_kl_loss_weight_epochs_idx = 0
    kl_loss_weight = kl_loss_weight_values[current_kl_loss_weight_epochs_idx]
    current_kl_loss_weight_epochs_idx += 1
    
    # TODO: check whether the param is appropriate (i.e., betas)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(kernel_size = 11, betas = (0.8,0.6),data_range=value_range).cuda()
    # lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()
    
    for epoch in range(epochs):
        if epoch % regenerate_sampled_points_freq == 0:
            # generate sampled coords and their corresponding target values
            sample_coords, target_values = regenerate_sampled_points(dataset_for_sampling)
        
        epoch = epoch + resume_epoch
        
        running_mae_loss = 0.0
        running_mse_loss = 0.0
        running_mse_loss_val = 0.0
        running_kl_loss = 0.0
        running_ms_ssim_loss = 0.0
        running_lpips_loss = 0.0
        running_geometry_loss = 0.0
        running_loss = 0.0
        total_elems = 0
        
        # fetch the correct kl_loss first
        # check to avoid index error
        if current_kl_loss_weight_epochs_idx < len(kl_loss_weight_epochs): 
            if epoch == kl_loss_weight_epochs[current_kl_loss_weight_epochs_idx]:
                kl_loss_weight = kl_loss_weight_values[current_kl_loss_weight_epochs_idx]
                current_kl_loss_weight_epochs_idx += 1
        
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
                    # if epoch == 1:
                    #     device = torch.device("cuda:0")  # or another index
                    #     print(torch.cuda.memory_allocated(device) / 1024 ** 2)
                    channel = 16
                    # for training on all volumes
                    if check_plane_idx in raw_data[1]:
                        for plane in range (3):
                            plot_single_channel(output[0][torch.where(raw_data[1] == check_plane_idx)[0].item()][plane][channel].detach(), f"epoch{epoch}_plane{plane}", save_path=os.path.join(run_dir, f"epoch{epoch}_plane{plane}.png"))
                    # for only one volume
                    # for plane in range (3):
                    #     plot_single_channel(output[0][0][plane][channel].detach(), f"epoch{epoch}_plane{plane}", save_path=os.path.join(run_dir, f"epoch{epoch}_plane{plane}.png"))
                    
                # reconstructed results is the first element of the output (output[0])
                mae_loss = mae_loss_weight * F.l1_loss(output[0], raw_data[0])
                mse_loss = F.mse_loss(output[0], raw_data[0])
                mse_loss_val = mse_loss.item()
                mse_loss = mse_loss_weight * mse_loss
                kl_loss = kl_loss_weight * vae_model.module.loss_function(*output)
                ms_ssim_loss = ms_ssim_loss_weight * (1 - ms_ssim(output[0], raw_data[0]))
                # lpips_loss = lpips_loss_weight * lpips(output[0].reshape([-1, 1, output[0].shape[3], output[0].shape[4]]).repeat(1, 3, 1, 1), raw_data[0].reshape([-1, 1, raw_data[0].shape[3], raw_data[0].shape[4]]).repeat(1, 3, 1, 1))
                # lpips_loss = lpips_loss_weight * lpips(output[0].reshape([-1, 1, output[0].shape[3], output[0].shape[4]]).expand(-1, 3, -1, -1), raw_data[0].reshape([-1, 1, raw_data[0].shape[3], raw_data[0].shape[4]]).expand(-1, 3, -1, -1))
                lpips_loss = torch.tensor(0).cuda()
                # load VAE-reconstructed triplanes into triplane models
                load_params_to_triplane(triplane, output[0], original_triplane_min, original_triplane_max, min, max)
                # if (epoch % ckpt_freq == (ckpt_freq - 1)) or (epoch == (epochs + resume_epoch) - 1):
                #     debug = True
                # else:
                #     debug = False
                debug = False
                geometry_loss = geometry_loss_weight * calculate_geometry_loss(net, triplane, raw_data[1], sample_coords, target_values, debug)
                loss = mae_loss + mse_loss + kl_loss + ms_ssim_loss + lpips_loss + geometry_loss
                # loss = recon_loss
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            running_mae_loss += mae_loss.item() * raw_data[0].shape[0]
            running_mse_loss += mse_loss.item() * raw_data[0].shape[0]
            running_mse_loss_val += mse_loss_val * raw_data[0].shape[0]
            running_kl_loss += kl_loss.item() * raw_data[0].shape[0]
            running_ms_ssim_loss += ms_ssim_loss.item() * raw_data[0].shape[0]
            running_lpips_loss += lpips_loss.item() * raw_data[0].shape[0]
            running_geometry_loss += geometry_loss.item() * raw_data[0].shape[0]
            running_loss += running_mae_loss + running_mse_loss + running_kl_loss + running_ms_ssim_loss + running_lpips_loss + running_geometry_loss
            total_elems += raw_data[0].shape[0]
            
            # TODO: need to sperate PSNR evaluation from each volume (cause currently has four volumes in one batch)
            # console_logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, KL loss: {kl_loss.item():0,.6f}, Reconstruction PSNR: {(20 * torch.log10(raw_data.max() - raw_data.min() / torch.sqrt(recon_loss))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
            if console_logger is not None:
                console_logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, MAE loss: {mae_loss.item():0,.6f}, MSE loss: {mse_loss.item():0,.6f}, KL loss: {kl_loss.item():0,.6f}, MS-SSIM loss: {ms_ssim_loss.item():0,.6f}, LPIPS loss: {lpips_loss.item():0,.6f}, Geometry loss: {geometry_loss.item():0,.6f}, Reconstruction PSNR: {(20 * np.log10(value_range / np.sqrt(mse_loss_val))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
                # console_logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, Reconstruction PSNR: {(20 * np.log10(value_range / np.sqrt(recon_loss.item()))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
            else:
                print(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, MAE loss: {mae_loss.item():0,.6f}, MSE loss: {mse_loss.item():0,.6f}, KL loss: {kl_loss.item():0,.6f}, MS-SSIM loss: {ms_ssim_loss.item():0,.6f}, LPIPS loss: {lpips_loss.item():0,.6f}, Geometry loss: {geometry_loss.item():0,.6f}, Reconstruction PSNR: {(20 * np.log10(value_range / np.sqrt(mse_loss_val))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
                # print(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, Reconstruction PSNR: {(20 * np.log10(value_range / np.sqrt(recon_loss.item()))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
            # import pdb; pdb.set_trace()
        
        # calculate the average loss for the epoch
        assert total_elems == len(train_dataloader.dataset), "total_elems should be equal to the dataset size"
        last_mae_loss = running_mae_loss / total_elems
        last_mse_loss = running_mse_loss / total_elems
        last_mse_loss_val = running_mse_loss_val / total_elems
        last_kl_loss = running_kl_loss / total_elems
        last_ms_ssim_loss = running_ms_ssim_loss / total_elems
        last_lpips_loss = running_lpips_loss / total_elems
        last_geometry_loss = running_geometry_loss / total_elems
        last_loss = running_loss / total_elems
        last_PSNR = 20 * np.log10(value_range / np.sqrt(last_mse_loss_val))
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("Loss/Train_MAE", last_mae_loss, epoch)
            tensorboard_writer.add_scalar("Loss/Train_MSE", last_mse_loss, epoch)
            tensorboard_writer.add_scalar("Loss/Train_KL", last_kl_loss, epoch)
            tensorboard_writer.add_scalar("Loss/Train_MS_SSIM", last_ms_ssim_loss, epoch)
            tensorboard_writer.add_scalar("Loss/Train_LPIPS", last_lpips_loss, epoch)
            tensorboard_writer.add_scalar("Loss/Train_GEO", last_geometry_loss, epoch)
            tensorboard_writer.add_scalar("Loss/Train", last_loss, epoch)
            tensorboard_writer.add_scalar("Loss/Train_PSNR", last_PSNR, epoch)
            tensorboard_writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)
        
        # adjust learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(last_loss)
        else:
            scheduler.step()  # StepLR, MultiStepLR, CosineAnnealingLR, etc.
        
        # save latent triplanes for inference later
        if (epoch % ckpt_freq == (ckpt_freq - 1)) or (epoch == (epochs + resume_epoch) - 1):
            outputs = []
            indices = []
            vae_model.eval()
            with torch.no_grad():
                for batch_idx, raw_data in enumerate(train_dataloader):
                    mu, log_var = vae_model.module.encode(raw_data[0])
                    output = vae_model.module.reparameterize(mu, log_var)
                    outputs.append(output)
                    indices.append(raw_data[1])        
            outputs = torch.cat(outputs, dim=0)
            indices = torch.cat(indices, dim=0)
            # Sort outputs by indices
            sorted_indices, sort_order = torch.sort(indices)
            outputs = outputs[sort_order]
            torch.save({"weights_latent_space": outputs}, os.path.join(run_dir, f"latent_triplanes_epoch_{epoch}.pt"))
            vae_model.train()
        
        # save the model at checkpoint
        if (epoch % ckpt_freq == (ckpt_freq - 1)) or (epoch == (epochs + resume_epoch) - 1):
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': last_loss,
                'mae_loss': last_mae_loss,
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
    loaded_model = torch.load(args.pretrained_triplane_file_path)
    triplane_weights = [loaded_model['triplane_state_dict'][f"{idx}.triplane"] for idx in range(n_timesteps)]
    triplane_weights = torch.cat(triplane_weights, dim=0)
    
    # normalize triplane value to -1, 1
    # normalization for training only one volume
    # triplane_weights[50:51][0] = (triplane_weights[50:51][0] - triplane_weights[50:51][0].min()) / (triplane_weights[50:51][0].max() - triplane_weights[50:51][0].min())
    # triplane_weights[50:51][0] = triplane_weights[50:51][0] * 2 - 1
    
    # normalization for training all volumes
    original_triplane_min = triplane_weights.min()
    original_triplane_max = triplane_weights.max()
    triplane_weights = (triplane_weights - original_triplane_min) / (original_triplane_max - original_triplane_min)
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
    if args.scheduler_type == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.lr_gamma)
    elif args.scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=args.lr_gamma, patience=args.patience)
    else:
        raise RuntimeError("Didn't specify the type of scheduler to use")
    
    # Check whether loss weights have been set from command line arguments
    if (args.mae_loss_weight == None) or (args.mse_loss_weight == None) or (args.ms_ssim_loss_weight == None) or (args.lpips_loss_weight == None) or (args.kl_loss_weight_values == None) or (args.kl_loss_weight_epochs == None) or (args.geometry_loss_weight == None):
        raise RuntimeError("Missing one of the loss weights, please set all of them every time (both training from scratch and resume training)")
    
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
        if "--init_lr" in sys.argv and "--lr_gamma" in sys.argv and "--milestones" in sys.argv:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.init_lr
            # for MultiStepLR resume training, set milestones epochs relative to resume points
            # E.g., if resuming from epoch 1000 and want LR decay at 1500, keep milestone as 500 (not 1500).
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.lr_gamma)
            console_logger.debug(f"Reset learning rate to {args.init_lr}, milestone epochs to {args.milestones}, learning rate decay factor to {args.lr_gamma}")
        elif "--init_lr" in sys.argv and "--lr_gamma" in sys.argv and "--patience" in sys.argv:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.init_lr
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=args.lr_gamma, patience=args.patience)
            console_logger.debug(f"Reset learning rate to {args.init_lr}, patience epochs to {args.patience}, learning rate decay factor to {args.lr_gamma}")
        elif "--init_lr" in sys.argv or "--lr_gamma" in sys.argv or "--patience" in sys.argv or "--milestones" in sys.argv:
            raise RuntimeError("Only reset either init_lr, lr_gamma, patience, or milestones for resuming training, please reset three arguments at the same time")
        else:
            scheduler.load_state_dict(loaded_ckpt["scheduler_state_dict"])
            console_logger.debug(f"Use original learning rate and scheduler settings")
        # free up memory for training
        del loaded_ckpt
        torch.cuda.empty_cache()
        console_logger.debug(f"MAE Loss Weight: {args.mae_loss_weight}, MSE Loss Weight: {args.mse_loss_weight}, MS SSIM Loss Weight: {args.ms_ssim_loss_weight}")
        console_logger.debug(f"LPIPS Loss Weight: {args.lpips_loss_weight}, KL Loss Weight Values:{args.kl_loss_weight_values}, KL Loss Weight Epochs:{args.kl_loss_weight_epochs}")
        console_logger.debug(f"Geometry Loss Weight: {args.geometry_loss_weight}")
        
    # training from scratch
    else:
        resume_epoch = 0
        tensorboard_writer.add_text("Model/Architecture", f"```\n{model_arch_str}\n```", global_step=0)
    
        console_logger.debug("Experiment description: " + args.description)
        console_logger.debug(f"Batch Size: {args.batch_size}, Epochs: {args.epochs}, Checkpoint Frequency: {args.ckpt_freq}")
        if args.milestones:
            console_logger.debug(f"Initial Learning rate: {args.init_lr}, Milestones: {args.milestones}, Learning rate Decay Factor: {args.lr_gamma}")
        else:
            console_logger.debug(f"Initial Learning rate: {args.init_lr}, Patience Epochs: {args.patience}, Learning rate Decay Factor: {args.lr_gamma}")
        console_logger.debug(f"Model config: autoencoder_config.triplane.{args.model_config}")
        console_logger.debug(f"MAE Loss Weight: {args.mae_loss_weight}, MSE Loss Weight: {args.mse_loss_weight}, MS SSIM Loss Weight: {args.ms_ssim_loss_weight}")
        console_logger.debug(f"LPIPS Loss Weight: {args.lpips_loss_weight}, KL Loss Weight Values:{args.kl_loss_weight_values}, KL Loss Weight Epochs:{args.kl_loss_weight_epochs}")
        console_logger.debug(f"Geometry Loss Weight: {args.geometry_loss_weight}")
        
    # prepare pre-sampled points' coordinates and values
    sample_batch_size = 2**10
    dataset_for_sampling = SampleTimevaryingDataset(
        raw_data_prefix="/home/kctung/vortices",
        raw_data_filename_without_timestep="vorts",
        file_ext="data",
        res=[128, 128, 128],
        n_timesteps=n_timesteps,
        n_channels=1,
        sample_batch_size=sample_batch_size
    )
    
    # instantiate triplane models and load pre-trained MLP
    triplane_config_path = "fit_triplane/base_timevarying.json"
    net, triplane = create_triplane_model(triplane_config_path, args.batch_size)
    net.load_state_dict(loaded_model['net_state_dict'])
    
    train_vae(vae_model, train_dataloader, optimizer, scheduler, args.epochs, tensorboard_writer, console_logger, run_dir, args.ckpt_freq, resume_epoch,
              args.mae_loss_weight, args.mse_loss_weight, args.ms_ssim_loss_weight, args.lpips_loss_weight, args.kl_loss_weight_values, args.kl_loss_weight_epochs,
              args.geometry_loss_weight, dataset_for_sampling, net, triplane, original_triplane_min, original_triplane_max)