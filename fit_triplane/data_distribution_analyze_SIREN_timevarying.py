from pysampler import create_sampler, decode_shadow, decode
from easydict import EasyDict as edict
import argparse
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import os, sys
import json
import matplotlib.pyplot as plt
from fit_shadow_subset_training_SIREN import NeurCompNet
from pathlib import Path

# Add parent directory to sys.path
# TODO: make it more flexible to call timevarying_data_helper anywhere
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from timevarying_data_helper import cartesian_to_spherical_coords, TimevaryingDataset_with_Sampler

# for debug
def only_decode_raw_shadow(sampler, data_res, chunk_size, tfn_file_path, angle=[0.5, 0.5]):
    targets = []
    for coord_chunk in generate_coords_chunks(data_res, chunk_size):
        target = torch.empty([coord_chunk.shape[0], 1]).float().cuda()
        decode_shadow(sampler, coord_chunk, target, angle, tfn_file_path)
        targets.append(target)
    
    targets = torch.cat(targets, dim=0)
    
    # targets.detach().cpu().numpy().astype(np.float32).tofile(f"test_shadow_volume.bin")
    return targets

def generate_coords_chunks(data_res, chunk_size, device='cuda'):
    """Yield chunks of coordinates from the full 3D grid."""
    gridz, gridy, gridx = torch.meshgrid(
        torch.linspace(0, 1, data_res[2]),  # z slowest
        torch.linspace(0, 1, data_res[1]),
        torch.linspace(0, 1, data_res[0]),  # x fastest
        indexing='ij'
    )
    # the accessing pattern in flattened volume: [1,0,0], [2,0,0], [3,0,0] ... (x change fastest)
    coords = torch.stack([gridx, gridy, gridz], dim=3).reshape(-1, 3)  # [N, 3]
    
    for start in range(0, coords.shape[0], chunk_size):
        end = start + chunk_size
        # allocate memory on CPU, only move to GPU when used for model inference
        yield coords[start:end].to(device)

def cal_GT_hist(n_instances, data_res, chunk_size, data_sampler):
    hist_cache = []
    with torch.no_grad():
        for batch_idx in range(n_instances):
            targets = []
            for coord_chunk in generate_coords_chunks(data_res, chunk_size):
                target = torch.empty([coord_chunk.shape[0], 1]).float().cuda()
                decode(data_sampler.get_sampler(batch_idx), coord_chunk, target)
                targets.append(target.cpu())
            targets = torch.cat(targets, dim=0)
            counts, bin_edges = np.histogram(targets.numpy(), bins=100)
            hist_cache.append((counts, bin_edges))
            
            # save the GPU memory 
            del targets
            torch.cuda.empty_cache()
            # HACK: delete sampler after use, otherwise, would encounter OOM error
            data_sampler.delete_sampler(batch_idx)
    return hist_cache

def inference(n_instances, data_res, chunk_size, value_range, nets, data_sampler, loaded_model, recon_type, all_timesteps):
    psnr_list = []
    hist_cache = []
    timestep_list = []
    with torch.no_grad():
        for batch_idx in range(n_instances):
            nets.load_state_dict(loaded_model['net_state_dict'])
            preds = []
            for coord_chunk in generate_coords_chunks(data_res, chunk_size):
                # preds.append(net(triplane[batch_idx](coord_chunk, 0)).cpu())
                # need to clamp the value range in case extreme outliers would make the rest of most values gather in one bin
                # preds.append(nets[batch_idx](coord_chunk).clamp(-1.0, 2.0).cpu())
                preds.append(nets[batch_idx](coord_chunk).cpu())
                # used when transforming the input values with inverse sigmoid
                # preds.append(torch.sigmoid(net(triplane[batch_idx](coord_chunk, 0))).cpu())
            # outputs = net(triplane[batch_idx](coords, 0))
            # outputs = outputs.view(raw_data.shape)
            outputs = torch.cat(preds, dim=0)
            # drop NaN
            indices_to_preserve = ~torch.isnan(outputs)
            outputs = outputs[indices_to_preserve]
            targets = []
            for coord_chunk in generate_coords_chunks(data_res, chunk_size):
                target = torch.empty([coord_chunk.shape[0], 1]).float().cuda()
                decode(data_sampler.get_sampler(batch_idx), coord_chunk, target)
                targets.append(target.cpu())
            targets = torch.cat(targets, dim=0)
            targets = targets[indices_to_preserve]
            # print("finish decoding")
            # Plot histogram
            # plt.hist(outputs.numpy(), bins=100, alpha=0.8, log=True, label=recon_type)
            # plt.hist(targets.numpy(), bins=100, color='red', alpha=0.8, log=True)
            counts, bin_edges = np.histogram(outputs.numpy(), bins=100)
            hist_cache.append((counts, bin_edges))
            
            loss = F.mse_loss(outputs, targets)
            PSNR = (20 * torch.log10(value_range / torch.sqrt(loss))).cpu()
            print("idx:", batch_idx, " psnr:", PSNR)
            psnr_list.append(PSNR)
            
            timestep_list.append(all_timesteps[batch_idx])
            
            # save the GPU memory 
            del outputs, targets, loss
            torch.cuda.empty_cache()
            # HACK: delete sampler after use, otherwise, would encounter OOM error
            data_sampler.delete_sampler(batch_idx)
    return psnr_list, hist_cache, timestep_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, nargs='+', default='base_timevarying.json')
    parser.add_argument('--SIREN_file_paths', type=str, nargs='+', default="../VAE_Reconstructed_triplane.pt")
    parser.add_argument('--SIREN_recon_types', type=str, nargs='+', default='vae_recon')
    parser.add_argument('--dims', type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--raw_data_dir', type=str, default="/media/data/qadwu/volume/vortices/vorts1.data")
    parser.add_argument('--output_activations', type=str, nargs='+') # default: None
    
    args = parser.parse_args()
    
    # data_res = [128, 128, 128]
    data_res = args.dims
    chunk_size = 65536*64

    config_files = args.configs
    SIREN_file_paths = args.SIREN_file_paths
    SIREN_recon_types = args.SIREN_recon_types
    output_activations = args.output_activations
    
    # pad config files with the last value if not enough config files are provided
    if len(config_files) < len(SIREN_file_paths):
        last = config_files[-1] if config_files else None
        config_files = config_files + [last] * (len(SIREN_file_paths) - len(config_files))
    # pad output_activations with the last value if not enough output_activations are provided
    if len(output_activations) < len(SIREN_file_paths):
        last = output_activations[-1] if output_activations else None
        output_activations = output_activations + [last] * (len(SIREN_file_paths) - len(output_activations))    
    
    psnr_lists = []
    hist_caches = []
    timestep_lists = []
    
    for config_file_path, SIREN_file_path, recon_type, output_activation in zip(config_files, SIREN_file_paths, SIREN_recon_types, output_activations):
        
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        config = edict(config)
        
        loaded_model = torch.load(SIREN_file_path, map_location="cpu")
        
        all_timesteps = loaded_model['timesteps']
        n_instances = len(all_timesteps)
        
        # instantiate multiple SIRENs (each instance has its own triplane)
        nets = [NeurCompNet(n_input_dims=3, 
                    n_output_dims=config.n_labels, bias=False, 
                    n_hidden_layers=config.n_layers, 
                    n_neurons=config.n_hid, is_residual=True).cuda() for _ in range(n_instances)]
        nets = nn.ModuleList(nets)
    
        data_sampler = TimevaryingDataset_with_Sampler(
                raw_data_dir=args.raw_data_dir,
                # HACK: just assume volume names start with "timestep_"
                raw_data_filename_without_timestep="timestep_",
                file_ext="bin",
                res=args.dims,
                data_type=args.dtype,
                n_instances=n_instances,
                n_channels=1,
                # NOTE: use the first timestep to preoptimize SIREN
                timesteps=all_timesteps,
                sample_batch_size=config.sample_batch_size,
            )
        value_range = data_sampler.value_range

        print(f"all timesteps: {all_timesteps}")

        psnr_list, hist_cache, timestep_list = inference(n_instances, data_res, chunk_size, value_range, nets, data_sampler, loaded_model, recon_type, all_timesteps)
        psnr_lists.append(psnr_list)
        hist_caches.append(hist_cache)
        timestep_lists.append(timestep_list)
    
    # use the light directions from the last loaded model
    # TODO: might need to find more reasonable impl. or just don't support varying length array
    GT_hist_cache = cal_GT_hist(n_instances, data_res, chunk_size, data_sampler)
    
    max_instances = max(len(lst) for lst in psnr_lists)
    for idx in range(max_instances):
        print(f"instance {idx} - ", end="")
        plt.figure(figsize=(6,4))
        for j in range(len(psnr_lists)):
            if idx < len(psnr_lists[j]):  # check if this list has enough elements
                print(f"{args.SIREN_recon_types[j]} PSNR: {psnr_lists[j][idx]}, ", end="")
                plt.hist(hist_caches[j][idx][1][:-1], hist_caches[j][idx][1], weights=hist_caches[j][idx][0], alpha=0.8, label=f"{args.SIREN_recon_types[j]} (PSNR: {psnr_lists[j][idx].item():0,.4f} / Timestep: {timestep_lists[j][idx]})", log=True)
            else:
                print(f"{args.SIREN_recon_types[j]} PSNR: N / A, ", end="")
        print("")
        plt.hist(GT_hist_cache[idx][1][:-1], GT_hist_cache[idx][1], weights=GT_hist_cache[idx][0], alpha=0.8, label="Ground Truth", log=True)
        plt.title(f"Value Dist of Reconstructed Timevarying Volume")
        # plt.title(f"Value Distribution of Reconstructed Shadow Coefficient Volume at instance {idx}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        # save to the first SIREN set's directory
        # HACK: I know this script can support plotting histogram from multiple SIREN sets,
        # but currently I mainly use for only 1 set to evaluate their histogram
        path = Path(SIREN_file_paths[0])
        dir_path = path.parent
        filename_without_extension = path.stem
        plt.savefig(os.path.join(dir_path, f"{filename_without_extension}_hist_ins_{idx}.png"))
        
        # plt.savefig(f"value_dist_pred_at_ins_{idx}.png")