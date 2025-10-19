import os
import torch
import torch.nn.functional as F
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import logging
import argparse
from datetime import datetime
import importlib

from autoencoder_2D_origin import VAE, VAE_no_KL
from timevarying_data_helper import LatentWeightDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Inference autoencoder on triplanes")
    parser.add_argument("--expdir", type=str, default="./logs/test_hashencoding_train/20250508-024915", help="Checkpoint Directory to load the model from")
    parser.add_argument("--model_file_name", type=str, default="vae_model_epoch_9999.ckpt", help="Model file name")
    parser.add_argument("--model_config", type=str, default="model_a", help="Model config file name")
    
    parser.add_argument("--pretrained_triplane_file_path", type=str, default="fit_triplane/ch_32_saved_model.ckpt", help="File Path to Pretrained Triplanes Model")
    return parser.parse_args()

if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    
    args = parse_args()
    
    # model configs are stored as python scripts, import the target config here
    cfg = importlib.import_module(f"autoencoder_config.triplane.{args.model_config}")
    
    vae_model = DDP(VAE_no_KL(cfg.vae_config, cfg.encoder_dims, cfg.feature_size_encoder, cfg.decoder_dims,
                        cfg.feature_size_decoder, cfg.fpn_encoders_layer_dim_idx, cfg.fpn_decoders_layer_dim_idx,
                        cfg.fpn_encoders_down_idx, cfg.fpn_encoders_up_idx, cfg.fpn_decoders_down_idx,
                        cfg.fpn_decoders_up_idx, cfg.block_config).cuda())
    # pretrained_vae_model = torch.load("logs/triplane_VAE_ch_32_single_volume/20250614-180226/vae_model_epoch_9999.ckpt")
    # pretrained_vae_model = torch.load("logs/triplane_AE_model_a/20250619-000758/vae_model_epoch_1399.ckpt")
    vae_model.load_state_dict(torch.load(os.path.join(args.expdir, args.model_file_name), weights_only=False)["model_state_dict"])
    
    # load pretrained tri-plane here
    pretrained_triplane_model = torch.load(args.pretrained_triplane_file_path, map_location="cpu")
    keys = pretrained_triplane_model['triplane_state_dict'].keys()
    n_instances = sum(1 for k in keys if k.endswith("triplane"))
    triplane_weights = [pretrained_triplane_model['triplane_state_dict'][f"{idx}.triplane"] for idx in range(n_instances)]
    triplane_weights = torch.cat(triplane_weights, dim=0)
    # normalization for training all volumes
    original_min = triplane_weights.min()
    original_max = triplane_weights.max()
    triplane_weights = (triplane_weights - original_min) / (original_max - original_min)
    triplane_weights = triplane_weights * 2 - 1
    min = -1
    max = 1
    
    train_dataloader = torch.utils.data.DataLoader(
        LatentWeightDataset(
        triplane_weights,
        cfg.vae_config["plane_shape"]),
        batch_size=1,
        shuffle=False)
    
    value_range = max - min
    # input_tensor = torch.randn(2, 3, 32, 128, 128).cuda()
    input_tensor = next(iter(train_dataloader))[0].cuda()
    out = vae_model(input_tensor)
    loss = vae_model.module.loss_function(*out)
    print("loss: {}".format(loss))
    print("z shape: {}".format(out[-1].shape))
    print("reconstruct shape: {}".format(out[0].shape))
    # samples = vae_model.module.sample(2)
    # print("samples shape: {}".format(samples[0].shape))

    # to make batchnorm function correctly for inference, need to call eval
    vae_model.eval()
    with torch.no_grad():
        for batch_idx, raw_data in enumerate(train_dataloader):
            
            raw_data[0] = raw_data[0].cuda()
            output = vae_model(raw_data[0])
            # reconstructed results is the first element of the output (output[0])
            recon_loss = F.mse_loss(output[0], raw_data[0])
            kl_loss = vae_model.module.loss_function(*output)
            loss = recon_loss + kl_loss
            # loss = recon_loss
            
            print(f"Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, KL loss: {kl_loss.item():0,.6f}, Reconstruction PSNR: {(20 * np.log10(value_range / np.sqrt(recon_loss.item()))):0,.4f}")
            # print(f"Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, Reconstruction PSNR: {(20 * np.log10(value_range / np.sqrt(recon_loss.item()))):0,.4f}")
            # import pdb; pdb.set_trace()
            # normalize output from -1~1 back to its original value range to align with the value range of triplane fitting
            output[0] = ((output[0] - min) / (max - min)) * (original_max - original_min) + original_min
            pretrained_triplane_model['triplane_state_dict'][f"{batch_idx}.triplane"] = output[0]
    
    torch.save(pretrained_triplane_model, os.path.join(args.expdir, f"VAE_Reconstructed_triplane.pt"))