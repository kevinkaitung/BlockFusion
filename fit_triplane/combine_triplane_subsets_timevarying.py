import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse

# NOTE: we didn't permute timevarying data
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # TODO: temporarily assume triplane path would be loaded in the order of the offsets, need to add code to check later
    parser.add_argument("--triplane_subsets_path", type=str, nargs="+")
    parser.add_argument("--mlp_path", type=str)
    parser.add_argument("--expdir", type=str)
    args = parser.parse_args()

    saved_model = dict()
    new_triplane_state_dict = dict()
    
    mlp_model = torch.load(args.mlp_path)
    # used to permute the triplanes back to its original order (the order before randomly permutation)
    # permuted_indices = mlp_model['permuted_indices']
    # inverse_indices = np.argsort(permuted_indices)

    for subset_path in args.triplane_subsets_path:
        subset_model = torch.load(subset_path)
        offset = subset_model['offset']
        triplane_state_dict = subset_model['triplane_state_dict']
        
        for key, value in triplane_state_dict.items():
            # Split into "<index>" and "<suffix>"
            idx, suffix = key.split(".", 1)
            idx_with_offset = int(idx) + offset
            # query the original index of this specific triplane and put it back to that index
            # new_idx = permuted_indices[idx_with_offset]
            new_idx = idx_with_offset
            new_key = f"{new_idx}.{suffix}"
            new_triplane_state_dict[new_key] = value
        
        print(f"Memory Allocate: {torch.cuda.memory_allocated() / (1024 ** 3)} / Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 3)}")
    
    saved_model['triplane_state_dict'] = new_triplane_state_dict
    saved_model['net_state_dict'] = mlp_model['net_state_dict']
    # NOTE: we didn't permute timestep instance before, probably no need this here
    saved_model['permuted_indices'] = mlp_model['permuted_indices']    

    # also permute the light direction
    # new_timesteps = []
    # for idx in inverse_indices:
    #     new_timesteps.append(mlp_model['timesteps'][idx])
    # saved_model['timesteps'] = new_timesteps
    saved_model['timesteps'] = mlp_model['timesteps']
    
    torch.save(saved_model, os.path.join(args.expdir, "pure_triplane_model_permuted.pt"))