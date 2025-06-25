import os
import torch
import torch.nn.functional as F
import numpy as np

import logging
import argparse
from datetime import datetime
import importlib

from autoencoder_2D_origin import VAE, VAE_no_KL
from timevarying_data_helper import LatentWeightDataset

# for debugging
import matplotlib.pyplot as plt
def plot_single_channel(data):
    plt.imshow(data.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title("2D Tensor Visualization")
    plt.savefig("test.png")
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Inference autoencoder on triplanes")
    parser.add_argument("--expdir", type=str, default="./logs/test_hashencoding_train/20250508-024915", help="Checkpoint Directory to load the model from")
    parser.add_argument("--model_file_name", type=str, default="vae_model_epoch_9999.ckpt", help="Model file name")
    parser.add_argument("--model_config", type=str, default="model_a", help="Model config file name")
    parser.add_argument("--encode_or_decode", type=str, default="encode", help="Extract latent if encode; map latent back to its original weights if decode")
    parser.add_argument("--diffusion_dir", type=str, default="")
    parser.add_argument("--diffusion_model_file_name", type=str, default="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.encode_or_decode == "encode":
        is_encode = True
    else:
        is_encode = False
    
    # model configs are stored as python scripts, import the target config here
    cfg = importlib.import_module(f"autoencoder_config.triplane.{args.model_config}")
    
    vae_model = torch.nn.DataParallel(VAE_no_KL(cfg.vae_config, cfg.encoder_dims, cfg.feature_size_encoder, cfg.decoder_dims,
                        cfg.feature_size_decoder, cfg.fpn_encoders_layer_dim_idx, cfg.fpn_decoders_layer_dim_idx,
                        cfg.fpn_encoders_down_idx, cfg.fpn_encoders_up_idx, cfg.fpn_decoders_down_idx,
                        cfg.fpn_decoders_up_idx, cfg.block_config)).cuda()
    # pretrained_vae_model = torch.load("logs/triplane_VAE_ch_32_single_volume/20250614-180226/vae_model_epoch_9999.ckpt")
    # pretrained_vae_model = torch.load("logs/triplane_AE_model_a/20250619-000758/vae_model_epoch_1399.ckpt")
    vae_model.load_state_dict(torch.load(os.path.join(args.expdir, args.model_file_name))["model_state_dict"])
    
    n_timesteps = 90
    
    # load pretrained tri-plane here
    pretrained_triplane_model = torch.load("fit_triplane/ch_32_saved_model.ckpt")
    triplane_weights = [pretrained_triplane_model['triplane_state_dict'][f"{idx}.triplane"] for idx in range(n_timesteps)]
    triplane_weights = torch.cat(triplane_weights, dim=0)
    # normalization for training all volumes
    original_min = triplane_weights.min()
    original_max = triplane_weights.max()
    triplane_weights = (triplane_weights - original_min) / (original_max - original_min)
    triplane_weights = triplane_weights * 2 - 1
    min = -1
    max = 1
    
    if not is_encode:
        # need to load diffusion-pretrained latent weights
        # TODO: make sure the key name aligns with the one generated in diffusion_inference.py
        latent_weights = torch.load(os.path.join(args.diffusion_dir, args.diffusion_model_file_name))["weights_latent_space"]
        
    
    if is_encode:
        dataset = LatentWeightDataset(
        triplane_weights,
        cfg.vae_config["plane_shape"])
    else:
        # three times at last dimension for 3 triplanes
        z_shape = [cfg.vae_config["z_shape"][0], cfg.vae_config["z_shape"][1], cfg.vae_config["z_shape"][2] * 3]
        dataset = LatentWeightDataset(
        latent_weights,
        z_shape)
    
    train_dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=1,
        shuffle=False)
    
    if is_encode:
        input_tensor = next(iter(train_dataloader))[0].cuda()
        out = vae_model(input_tensor)
        # loss = vae_model.loss_function(*out)
        # print("loss: {}".format(loss))
        print("z shape: {}".format(out[-1].shape))
        print("reconstruct shape: {}".format(out[0].shape))
        # samples = vae_model.sample(2)
        # print("samples shape: {}".format(samples[0].shape))
    
    outputs = []
    # to make batchnorm function correctly for inference, need to call eval
    vae_model.eval()
    with torch.no_grad():
        for batch_idx, raw_data in enumerate(train_dataloader):
            
            if is_encode:
                output = vae_model.module.encode(raw_data[0])
            else:
                # TODO: check the data shape of raw_data when decoding
                output = vae_model.module.decode(raw_data[0])
                output = ((output - min) / (max - min)) * (original_max - original_min) + original_min
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)
    
    if is_encode:
        torch.save({"weights_latent_space": outputs}, os.path.join(args.expdir, "latent_triplanes.pt"))
    else:
        for idx in range(n_timesteps):
            # since we only use one diffusion-generated sample, copy it for all timesteps
            # replace original triplanes with the newly generated ones, and store as new triplanes
            # TODO:
            # pretrained_triplane_model['triplane_state_dict'][f"{idx}.triplane"] = outputs[idx % 8:idx % 8 + 1]
            pretrained_triplane_model['triplane_state_dict'][f"{idx}.triplane"] = outputs[:1]
        torch.save(pretrained_triplane_model, os.path.join(args.diffusion_dir, "Diffusion_VAE_Reconstructed_triplane.pt"))