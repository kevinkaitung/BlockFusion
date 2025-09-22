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
    parser.add_argument("--diffusion_latent_triplane_file_name", type=str, default="")
    
    parser.add_argument("--pretrained_triplane_file_path", type=str, default="fit_triplane/ch_32_saved_model.ckpt", help="File Path to Pretrained Triplanes Model")
    return parser.parse_args()

if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")

    args = parse_args()
    
    if args.encode_or_decode == "encode":
        is_encode = True
    else:
        is_encode = False
    
    # model configs are stored as python scripts, import the target config here
    cfg = importlib.import_module(f"autoencoder_config.triplane.{args.model_config}")
    
    # currently, expect using only one GPU for inference
    vae_model = DDP(VAE_no_KL(cfg.vae_config, cfg.encoder_dims, cfg.feature_size_encoder, cfg.decoder_dims,
                        cfg.feature_size_decoder, cfg.fpn_encoders_layer_dim_idx, cfg.fpn_decoders_layer_dim_idx,
                        cfg.fpn_encoders_down_idx, cfg.fpn_encoders_up_idx, cfg.fpn_decoders_down_idx,
                        cfg.fpn_decoders_up_idx, cfg.block_config).cuda())
    # pretrained_vae_model = torch.load("logs/triplane_VAE_ch_32_single_volume/20250614-180226/vae_model_epoch_9999.ckpt")
    # pretrained_vae_model = torch.load("logs/triplane_AE_model_a/20250619-000758/vae_model_epoch_1399.ckpt")
    vae_model.load_state_dict(torch.load(os.path.join(args.expdir, args.model_file_name), weights_only=False)["model_state_dict"])
        
    # load pretrained tri-plane here
    pretrained_triplane_model = torch.load(args.pretrained_triplane_file_path)
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
    
    if not is_encode:
        # need to load diffusion-pretrained latent weights
        latent_triplane_model = torch.load(os.path.join(args.diffusion_dir, args.diffusion_latent_triplane_file_name))
        latent_weights = latent_triplane_model["weights_latent_space"]
        light_dir_cartesian = latent_triplane_model["light_dir_cartesian"]
    else:
        pretrained_triplane_light_dir_cartesian = pretrained_triplane_model['light_dir_cartesian']
        
    
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
                mu, log_var = vae_model.module.encode(raw_data[0])
                output = vae_model.module.reparameterize(mu, log_var)
                # output = vae_model.module.encode(raw_data[0])
            else:
                # TODO: check the data shape of raw_data when decoding
                output = vae_model.module.decode(raw_data[0])
                output = ((output - min) / (max - min)) * (original_max - original_min) + original_min
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)
        print("outputs shape: {}".format(outputs.shape))
    
    if is_encode:
        torch.save({"weights_latent_space": outputs,
                    "light_dir_cartesian":pretrained_triplane_light_dir_cartesian
                    }, os.path.join(args.expdir, "latent_triplanes.pt"))
    else:
        output_triplane_model = {
            'net_state_dict':pretrained_triplane_model['net_state_dict'],
            'light_dir_cartesian':light_dir_cartesian,
            'triplane_state_dict': {},
        }
        for idx in range(latent_weights.shape[0]):
            # pretrained_triplane_model['triplane_state_dict'][f"{idx}.triplane"] = outputs[idx % 8:idx % 8 + 1]
            output_triplane_model['triplane_state_dict'][f"{idx}.triplane"] = outputs[idx:idx+1]
            output_triplane_model['triplane_state_dict'][f"{idx}.plane_axes"] = pretrained_triplane_model["triplane_state_dict"]["0.plane_axes"]
            
        torch.save(output_triplane_model, os.path.join(args.diffusion_dir, "Diffusion_VAE_Reconstructed_triplane.pt"))