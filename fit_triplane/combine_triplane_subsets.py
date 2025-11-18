import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # TODO: temporarily assume triplane path would be loaded in the order of the offsets, need to add code to check later
    parser.add_argument("--triplane_subsets_path", type=str, nargs="+")
    parser.add_argument("--mlp_path", type=str)
    parser.add_argument("--expdir", type=str)
    args = parser.parse_args()

    saved_model = dict()
    new_triplane_state_dict = dict()

    for subset_path in args.triplane_subsets_path:
        subset_model = torch.load(subset_path)
        offset = subset_model['offset']
        triplane_state_dict = subset_model['triplane_state_dict']
        
        for key, value in triplane_state_dict.items():
            # Split into "<index>" and "<suffix>"
            idx, suffix = key.split(".", 1)
            new_idx = int(idx) + offset
            new_key = f"{new_idx}.{suffix}"
            new_triplane_state_dict[new_key] = value
        
        print(f"Memory Allocate: {torch.cuda.memory_allocated() / (1024 ** 3)} / Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 3)}")
    
    mlp_model = torch.load(args.mlp_path)
    saved_model['triplane_state_dict'] = new_triplane_state_dict
    saved_model['net_state_dict'] = mlp_model['net_state_dict']
    saved_model['light_dir_cartesian'] = mlp_model['light_dir_cartesian']
    saved_model['light_dir_spherical'] = mlp_model['light_dir_spherical']
    saved_model['permuted_indices'] = mlp_model['permuted_indices']
    
    torch.save(saved_model, os.path.join(args.expdir, "pure_triplane_model.pt"))