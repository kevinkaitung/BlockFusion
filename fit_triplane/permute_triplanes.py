import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--triplane_path", type=str)
    parser.add_argument("--mlp_path", type=str)
    parser.add_argument("--expdir", type=str)
    args = parser.parse_args()

    saved_model = dict()
    new_triplane_state_dict = dict()
    
    mlp_model = torch.load(args.mlp_path)
    triplane_model = torch.load(args.triplane_path)
    old_triplane_state_dict = triplane_model["triplane_state_dict"]
    # used to permute the triplanes back to its original order (the order before randomly permutation)
    permuted_indices = mlp_model['permuted_indices']
    inverse_indices = np.argsort(permuted_indices)

    n_instances = len(permuted_indices)

    for idx in range(n_instances):
        new_idx = permuted_indices[idx]
        new_triplane_state_dict[f'{new_idx}.triplane'] = old_triplane_state_dict[f'{idx}.triplane']
        new_triplane_state_dict[f'{new_idx}.plane_axes'] = old_triplane_state_dict[f'{idx}.plane_axes']
    
    saved_model['triplane_state_dict'] = new_triplane_state_dict
    saved_model['net_state_dict'] = mlp_model['net_state_dict']
    saved_model['permuted_indices'] = mlp_model['permuted_indices']    

    # also permute the light direction
    new_light_dir_cartesian = []
    new_light_dir_spherical = []
    for idx in inverse_indices:
        new_light_dir_cartesian.append(mlp_model['light_dir_cartesian'][idx])
        new_light_dir_spherical.append(mlp_model['light_dir_spherical'][idx])
    saved_model['light_dir_cartesian'] = new_light_dir_cartesian
    saved_model['light_dir_spherical'] = new_light_dir_spherical
    
    torch.save(saved_model, os.path.join(args.expdir, "pure_triplane_model_permuted.pt"))