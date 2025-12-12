import torch
import argparse
import os
import numpy as np
from visualize_triplane import plot_single_channel

check_plane_idx = 40

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str)
parser.add_argument("--filename", type=str)
args = parser.parse_args()

# load pretrained tri-plane here
loaded_model = torch.load(os.path.join(args.model_dir, args.filename), map_location="cpu")
keys = loaded_model['triplane_state_dict'].keys()
n_instances = sum(1 for k in keys if k.endswith("triplane"))

upsampler = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
for idx in range(n_instances):
    loaded_model['triplane_state_dict'][f"{idx}.triplane"] = torch.unsqueeze(upsampler(torch.squeeze(loaded_model['triplane_state_dict'][f"{idx}.triplane"].cuda(), 0)), 0).cpu()
    # plot one of the upsampled triplane images
    if idx == check_plane_idx:
        for dim in range(3):
            plot_single_channel(
                loaded_model['triplane_state_dict'][f"{idx}.triplane"][0][dim][16].detach(), 
                    title=f"plane_offset_0_dim_{dim}_upsampled",
                    save_path=os.path.join(args.model_dir, f"plane_offset_0_dim_{dim}_upsampled.png")
            )

# save upsampled triplanes as new model
torch.save(loaded_model, os.path.join(args.model_dir, f"pure_triplane_model_upsampled.pt"))