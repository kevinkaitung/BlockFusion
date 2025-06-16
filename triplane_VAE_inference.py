import os
import torch
import torch.nn.functional as F
import numpy as np

import logging
import argparse
from datetime import datetime

from autoencoder_2D_origin import VAE, VAE_no_KL
from timevarying_data_helper import LatentWeightDataset

if __name__ == "__main__":
    vae_config = {"kl_std": 0.25,
                  "kl_weight": 0.001,
                  "plane_shape": [3, 32, 128, 128],
                  "z_shape": [4, 32, 32],
                  "num_heads": 16,
                  "transform_depth": 1}
    # TODO: should receive arguments to specify the location of pretrained model and other arguments
    vae_model = VAE_no_KL(vae_config).cuda()
    # pretrained_vae_model = torch.load("logs/triplane_VAE_ch_32_single_volume/20250614-180226/vae_model_epoch_9999.ckpt")
    pretrained_vae_model = torch.load("logs/triplane_revised_autoencoder_training/20250615-004012/vae_model_epoch_2499.ckpt")
    vae_model.load_state_dict(pretrained_vae_model['model_state_dict'])
    
    n_timesteps = 90
    
    # load pretrained tri-plane here
    loaded_model = torch.load("fit_triplane/ch_32_saved_model.ckpt")
    triplane_weights = [loaded_model['triplane_state_dict'][f"{idx}.triplane"] for idx in range(n_timesteps)]
    triplane_weights = torch.cat(triplane_weights, dim=0)
    # TODO: check whether it makes sense to pass vae_config["plane_shape"] into LatentWeightDataset
    # since it is originally designed to receive latent weights in 3D space (hash grid)
    # Need to check how does the original VAE receive input:
    # [batch size, 3 (triplanes), #channels, res, res] or [batch size, #channels, res, res * 3 (triplanes)]
    # seems both work(?)
    # normalize values to -1~1 to align with the value range used in training (for only one volume)
    # original_max = triplane_weights[50:51][0].max()
    # original_min = triplane_weights[50:51][0].min()
    # triplane_weights[50:51][0] = (triplane_weights[50:51][0] - original_min) / (original_max - original_min)
    # triplane_weights[50:51][0] = triplane_weights[50:51][0] * 2 - 1
    # normalization for training all volumes
    original_min = triplane_weights.min()
    original_max = triplane_weights.max()
    triplane_weights = (triplane_weights - original_min) / (original_max - original_min)
    triplane_weights = triplane_weights * 2 - 1
    
    train_dataloader = torch.utils.data.DataLoader(
        LatentWeightDataset(
        triplane_weights,
        vae_config["plane_shape"]),
        batch_size=1,
        shuffle=False)
    min, max = train_dataloader.dataset.get_value_range()
    value_range = max - min
    # input_tensor = torch.randn(2, 3, 32, 128, 128).cuda()
    input_tensor = next(iter(train_dataloader))[0].cuda()
    out = vae_model(input_tensor)
    # loss = vae_model.loss_function(*out)
    # print("loss: {}".format(loss))
    print("z shape: {}".format(out[-1].shape))
    print("reconstruct shape: {}".format(out[0].shape))
    # samples = vae_model.sample(2)
    # print("samples shape: {}".format(samples[0].shape))
    # to make batchnorm function correctly for inference, need to call eval
    vae_model.eval()
    with torch.no_grad():
        for batch_idx, raw_data in enumerate(train_dataloader):
            
            output = vae_model(raw_data[0])
            # reconstructed results is the first element of the output (output[0])
            recon_loss = F.mse_loss(output[0], raw_data[0])
            # kl_loss = vae_model.loss_function(*output)
            # loss = recon_loss + kl_loss
            loss = recon_loss
            
            # print(f"Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, KL loss: {kl_loss.item():0,.6f}, Reconstruction PSNR: {(20 * np.log10(value_range / np.sqrt(recon_loss.item()))):0,.4f}")
            print(f"Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, Reconstruction PSNR: {(20 * np.log10(value_range / np.sqrt(recon_loss.item()))):0,.4f}")
            # import pdb; pdb.set_trace()
            # normalize output from -1~1 back to its original value range to align with the value range of triplane fitting
            output[0] = ((output[0] - min) / (max - min)) * (original_max - original_min) + original_min
            loaded_model['triplane_state_dict'][f"{batch_idx}.triplane"] = output[0]
    
    torch.save(loaded_model, "VAE_Reconstructed_triplane_ch_32.pt")