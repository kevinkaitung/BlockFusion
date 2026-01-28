from easydict import EasyDict as edict
import argparse
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import os, sys
import json
from fit import Triplane, Network
import matplotlib.pyplot as plt
from pysampler import create_sampler, decode_shadow
from fit_shadow_randomly_generate import MLP_TCNN

# Add parent directory to sys.path
# TODO: make it more flexible to call timevarying_data_helper anywhere
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from timevarying_data_helper import cartesian_to_spherical_coords

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

def cal_GT_hist(n_instances, data_res, chunk_size, light_dirs, tfn_file_path):
    hist_cache = []
    with torch.no_grad():
        for batch_idx in range(n_instances):
            targets = []
            for coord_chunk in generate_coords_chunks(data_res, chunk_size):
                target = torch.empty([coord_chunk.shape[0], 1]).float().cuda()
                decode_shadow(sampler, coord_chunk, target, light_dirs[batch_idx], tfn_file_path)
                targets.append(target.cpu())
            targets = torch.cat(targets, dim=0)
            counts, bin_edges = np.histogram(targets.numpy(), bins=100)
            hist_cache.append((counts, bin_edges))
            
            # save the GPU memory 
            del targets
            torch.cuda.empty_cache()
    return hist_cache

def inference(n_instances, data_res, chunk_size, value_range, triplane, net, light_dirs, tfn_file_path, loaded_model, recon_type):
    psnr_list = []
    hist_cache = []
    with torch.no_grad():
        for batch_idx in range(n_instances):
            net.load_state_dict(loaded_model['net_state_dict'])
            triplane.load_state_dict(loaded_model['triplane_state_dict'])
            preds = []
            for coord_chunk in generate_coords_chunks(data_res, chunk_size):
                # preds.append(net(triplane[batch_idx](coord_chunk, 0)).cpu())
                # need to clamp the value range in case extreme outliers would make the rest of most values gather in one bin
                # preds.append(net(triplane[batch_idx](coord_chunk, 0)).clamp(-1.0, 2.0).cpu())
                preds.append(net(triplane[batch_idx](coord_chunk, 0)).cpu())
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
                decode_shadow(sampler, coord_chunk, target, light_dirs[batch_idx], tfn_file_path)
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
            
            # save the GPU memory 
            del outputs, targets, loss
            torch.cuda.empty_cache()
    return psnr_list, hist_cache

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, nargs='+', default='base_timevarying.json')
    parser.add_argument('--triplane_file_paths', type=str, nargs='+', default="../VAE_Reconstructed_triplane.pt")
    parser.add_argument('--triplane_recon_types', type=str, nargs='+', default='vae_recon')
    parser.add_argument('--dims', type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--raw_data_file_path', type=str, default="/media/data/qadwu/volume/vortices/vorts1.data")
    parser.add_argument('--tfn_file_path', type=str, default="/home/kctung/Projects/instant-vnr-pytorch/bindings/ovr/data/configs/vorts_shadow.json")
    parser.add_argument('--output_activations', type=str, nargs='+') # default: None
    
    args = parser.parse_args()
    
    # data_res = [128, 128, 128]
    data_res = args.dims
    chunk_size = 65536*192

    config_files = args.configs
    triplane_file_paths = args.triplane_file_paths
    triplane_recon_types = args.triplane_recon_types
    output_activations = args.output_activations
    
    # pad config files with the last value if not enough config files are provided
    if len(config_files) < len(triplane_file_paths):
        last = config_files[-1] if config_files else None
        config_files = config_files + [last] * (len(triplane_file_paths) - len(config_files))
    # pad output_activations with the last value if not enough output_activations are provided
    if len(output_activations) < len(triplane_file_paths):
        last = output_activations[-1] if output_activations else None
        output_activations = output_activations + [last] * (len(triplane_file_paths) - len(output_activations))    
    
    # volume reconstructed by triplane should between 0~1
    value_range = 1.0
    sampler = create_sampler("structuredRegular", "cuda", dims=args.dims, dtype=args.dtype, n_channels=1, filename=args.raw_data_file_path)
    
    psnr_lists = []
    hist_caches = []
    
    for config_file_path, triplane_file_path, recon_type, output_activation in zip(config_files, triplane_file_paths, triplane_recon_types, output_activations):
        
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        config = edict(config)
        
        loaded_model = torch.load(triplane_file_path, map_location="cpu")
        
        n_instances = len(loaded_model['light_dir_cartesian'])
        
        # net = Network(
        #     d_in=config.channel,
        #     d_hid=config.n_hid,
        #     n_layers=config.n_layers,
        #     d_out=config.n_labels,
        #     init_type="geo_init",
        # ).cuda()
        net = MLP_TCNN(n_input_dims=config.channel, n_output_dims=config.n_labels,
                n_hidden_layers=config.n_layers, n_neurons=config.n_hid,
                activation="ReLU", output_activation=output_activation)

        # instantiate multiple triplanes (each instance has its own triplane)
        triplane = [Triplane(
            reso=config.resolution,
            channel=config.channel,
            init_type="geo_init",
            objname=None,
        ).cuda() for _ in range(n_instances)]
        triplane = nn.ModuleList(triplane)
    
        light_dirs = cartesian_to_spherical_coords(np.array(loaded_model['light_dir_cartesian']))
        print(f"light dirs 1: {light_dirs}")
    
        # normalize the spherical coordinates to 0~1 (to comply with shadow sampler)
        light_dirs[:,0] = (light_dirs[:,0] % (2*np.pi)) / (2*np.pi)
        light_dirs[:,1] = light_dirs[:,1] / np.pi

        psnr_list, hist_cache = inference(n_instances, data_res, chunk_size, value_range, triplane, net, light_dirs, args.tfn_file_path, loaded_model, recon_type)
        psnr_lists.append(psnr_list)
        hist_caches.append(hist_cache)
    
    # use the light directions from the last loaded model
    # TODO: might need to find more reasonable impl. or just don't support varying length array
    GT_hist_cache = cal_GT_hist(n_instances, data_res, chunk_size, light_dirs, args.tfn_file_path)
    
    max_instances = max(len(lst) for lst in psnr_lists)
    for idx in range(max_instances):
        print(f"instance {idx} - ", end="")
        plt.figure(figsize=(6,4))
        for j in range(len(psnr_lists)):
            if idx < len(psnr_lists[j]):  # check if this list has enough elements
                print(f"{args.triplane_recon_types[j]} PSNR: {psnr_lists[j][idx]}, ", end="")
                plt.hist(hist_caches[j][idx][1][:-1], hist_caches[j][idx][1], weights=hist_caches[j][idx][0], alpha=0.8, label=f"{args.triplane_recon_types[j]} (PSNR: {psnr_lists[j][idx].item():0,.4f})", log=True)
            else:
                print(f"{args.triplane_recon_types[j]} PSNR: N / A, ", end="")
        print("")
        plt.hist(GT_hist_cache[idx][1][:-1], GT_hist_cache[idx][1], weights=GT_hist_cache[idx][0], alpha=0.8, label="Ground Truth", log=True)
        plt.title(f"Value Dist of Reconstructed Shadow Coefficient Volume")
        # plt.title(f"Value Distribution of Reconstructed Shadow Coefficient Volume at instance {idx}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.savefig(f"value_dist_pred_at_ins_{idx}.png")