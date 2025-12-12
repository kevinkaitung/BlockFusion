import argparse
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import os, sys
import json
import matplotlib.pyplot as plt
from pysampler import create_sampler, decode_shadow
from data_distribution_analyze import generate_coords_chunks
from datetime import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from timevarying_data_helper import cartesian_to_spherical_coords

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--raw_data_file_path', type=str, default="/media/data/qadwu/volume/vortices/vorts1.data")
    parser.add_argument('--tfn_file_path', type=str, default="/home/kctung/Projects/instant-vnr-pytorch/bindings/ovr/data/configs/vorts_shadow.json")
    parser.add_argument('--light_dir_file_path', type=str, default="../VAE_Reconstructed_triplane.pt")
    
    args = parser.parse_args()

    sampler = create_sampler("structuredRegular", "cuda", dims=args.dims, dtype=args.dtype, n_channels=1, filename=args.raw_data_file_path)
    
    try:
        with open(args.light_dir_file_path, 'r') as f:
            loaded_file = json.load(f)
            light_dirs = cartesian_to_spherical_coords(np.array(loaded_file['light_dir_cartesian']))
            print(f"loaded light directions (spherical coords): {light_dirs}")
            # normalize the spherical coordinates to 0~1 (to comply with shadow sampler)
            light_dirs[:,0] = (light_dirs[:,0] % (2*np.pi)) / (2*np.pi)
            light_dirs[:,1] = light_dirs[:,1] / np.pi
    except FileNotFoundError:
        print("Error: 'example.json' not found.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in 'example.json'.")
    
    n_instances = len(light_dirs)
    data_res = args.dims
    chunk_size = 65536*192

    # create directory to save the results
    dir_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(dir_name, exist_ok=True)

    with torch.no_grad():
        for batch_idx in range(n_instances):
            targets = []
            for coord_chunk in generate_coords_chunks(data_res, chunk_size):
                target = torch.empty([coord_chunk.shape[0], 1]).float().cuda()
                decode_shadow(sampler, coord_chunk, target, light_dirs[batch_idx], args.tfn_file_path)
                targets.append(target.cpu())
            targets = torch.cat(targets, dim=0)
            # save the volume
            # targets.detach().cpu().numpy().astype(np.float32).tofile(os.path.join(dir_name, f"shadow_volume_at_instance_{batch_idx}.bin"))
            
            # Plot histogram
            plt.figure(figsize=(6,4))
            counts, bin_edges = np.histogram(targets.numpy(), bins=100)
            plt.hist(bin_edges[:-1], bin_edges, weights=counts, alpha=0.8, label="Ground Truth Shadow", log=True)
            plt.title(f"Value Dist of Shadow Volume at light dir {light_dirs[batch_idx]}")
            # plt.title(f"Value Distribution of Reconstructed Shadow Coefficient Volume at instance {idx}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.savefig(os.path.join(dir_name, f"value_dist_GT_at_ins_{batch_idx}.png"))
               
            # save the GPU memory
            del targets
            torch.cuda.empty_cache()