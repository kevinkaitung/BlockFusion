from pysampler import create_sampler, decode_shadow, decode
import argparse
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import os, sys
import json
import matplotlib.pyplot as plt
from data_distribution_analyze import generate_coords_chunks
from datetime import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from timevarying_data_helper import TimevaryingDataset_with_Sampler

def histogram_equalization(vol):
    """
    vol: 2D numpy array (uint8), grayscale image in range [0,255]
    returns: equalized image
    """
    num_bins = 256
    # 1. Compute histogram (256 bins for 0-255)
    counts, bin_edges = np.histogram(vol, bins=num_bins)

    # 2. Compute cumulative distribution function (CDF)
    cdf = np.cumsum(counts)

    # 3. Normalize CDF to range [0,255]
    cdf_min = cdf[cdf > 0][0]   # first non-zero value
    # import pdb; pdb.set_trace()
    total_pixels = vol.numel()

    cdf_normalized = (cdf - cdf_min) / (total_pixels - cdf_min)
    vol = (vol - vol.min()) / (vol.max() - vol.min()) * (num_bins - 1)
    vol = vol.int()
    cdf_scaled = cdf_normalized
    # cdf_scaled = (cdf_normalized * 255).astype(np.uint8)
    # import pdb; pdb.set_trace()
    # 4. Map original image through lookup table
    equalized_vol = cdf_scaled[vol]

    return equalized_vol

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', type=int, nargs=3, default=[512, 512, 512])
    parser.add_argument('--dtype', type=str, default='uint8')
    parser.add_argument('--raw_data_dir', type=str, default="/eagle/ideas/AI_VIS/kctung/012_00215_055")
    
    args = parser.parse_args()

    # HACK: hard code the timevolumes info here
    timevolume_filename_stem = "timestep_"
    timevolume_filename_ext = "bin"
    all_timesteps = [idx for idx in range(600)]

    n_instances = len(all_timesteps)
    
    dataset = TimevaryingDataset_with_Sampler(
            raw_data_dir=args.raw_data_dir,
            # HACK: just assume volume names start with "timestep_"
            raw_data_filename_without_timestep=timevolume_filename_stem,
            file_ext=timevolume_filename_ext,
            res=args.dims,
            data_type=args.dtype,
            n_instances=n_instances,
            n_channels=1,
            # NOTE: use the first timestep to preoptimize SIREN
            timesteps=all_timesteps
    )
    
    data_res = args.dims
    # chunk_size = 65536*512
    # NOTE: volume res and itself aren't too big -> doable to generate coord grid at once
    chunk_size = data_res[0] * data_res[1] * data_res[2] * 3

    # create directory to save the results
    base_dir = "../logs"
    os.makedirs(base_dir, exist_ok=True)
    current_datatime = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = os.path.join(base_dir, f"Timevolumes_histogram_{current_datatime}")
    os.makedirs(dir_name, exist_ok=True)

    # create dir to save equalized volumes
    equalize_volume_dir = os.path.join(args.raw_data_dir, "equalized")
    os.makedirs(equalize_volume_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx in range(n_instances):
            print(f"Processing batch index: {batch_idx} / timestep: {all_timesteps[batch_idx]}")
            print(f"max mem allocated: {torch.cuda.max_memory_allocated() /10**9} GB")
            print(f"max mem reserved: {torch.cuda.max_memory_reserved() /10**9} GB")
            targets = []
            for coord_chunk in generate_coords_chunks(data_res, chunk_size, "cuda"):
                target = torch.empty([coord_chunk.shape[0], 1]).float().cuda()
                sampler = dataset.get_sampler(batch_idx)
                decode(sampler, coord_chunk, target)
                targets.append(target.cpu())
            targets = torch.cat(targets, dim=0)
            dataset.delete_sampler(batch_idx)
            
            # histogram equalize the volumes
            targets = histogram_equalization(targets)
            # NOTE: returned targets would become numpy array
            # HACK: because the data we deal with now is uint8, I just normalize it back to 0~255 for storing it as uint8
            targets *= 255
            # save the volume (convert back)
            targets.astype(args.dtype).tofile(os.path.join(equalize_volume_dir, f"{timevolume_filename_stem}{all_timesteps[batch_idx]}.{timevolume_filename_ext}"))
            
            # Plot histogram
            plt.figure(figsize=(6,4))
            counts, bin_edges = np.histogram(targets, bins=256)
            plt.hist(bin_edges[:-1], bin_edges, weights=counts, alpha=0.8, label="Ground Truth Timevolumes", log=True)
            plt.title(f"Value Dist of Timevolume at timestep {all_timesteps[batch_idx]}")
            # plt.title(f"Value Distribution of Reconstructed Shadow Coefficient Volume at instance {idx}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.savefig(os.path.join(dir_name, f"value_dist_GT_at_ins_{batch_idx}.png"))
            plt.close()