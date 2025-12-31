import torch
import numpy as np
from pysampler import create_sampler, decode_shadow
import argparse
from data_distribution_analyze import generate_coords_chunks
import json, os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from timevarying_data_helper import cartesian_to_spherical_coords

def calculate_dx(grid: torch.tensor):
    # grid: expect 3D tensor representing the values on the regular grid
    # TODO: might also need to consider receiving a list of values (put values sequentially not like a grid)
    
    dx = torch.zeros_like(grid)
    h = 1.0

    # forward difference (calculate for grid idx [0])
    dx[:, :, 0] = (grid[:, :, 1] - grid[:, :, 0]) / h

    # central difference (calculate for grid idx [1] ~ [-2])
    dx[:, :, 1:-1] = (grid[:, :, 2:] - grid[:, :, :-2]) / (2 * h)
    
    # backward difference (calculate for grid idx [-1])
    dx[:, :, -1] = (grid[:, :, -1] - grid[:, :, -2]) / h

    return dx
    
def calculate_dy(grid: torch.tensor):
    # grid: expect 3D tensor representing the values on the regular grid
    # TODO: might also need to consider receiving a list of values (put values sequentially not like a grid)
    
    dy = torch.zeros_like(grid)
    h = 1.0

    # forward difference (calculate for grid idx [0])
    dy[:, 0, :] = (grid[:, 1, :] - grid[:, 0, :]) / h

    # central difference (calculate for grid idx [1] ~ [-2])
    dy[:, 1:-1, :] = (grid[:, 2:, :] - grid[:, :-2, :]) / (2 * h)
    
    # backward difference (calculate for grid idx [-1])
    dy[:, -1, :] = (grid[:, -1, :] - grid[:, -2, :]) / h

    return dy

def calculate_dz(grid: torch.tensor):
    # grid: expect 3D tensor representing the values on the regular grid
    # TODO: might also need to consider receiving a list of values (put values sequentially not like a grid)
    
    dz = torch.zeros_like(grid)
    h = 1.0

    # forward difference (calculate for grid idx [0])
    dz[0, :, :] = (grid[1, :, :] - grid[0, :, :]) / h

    # central difference (calculate for grid idx [1] ~ [-2])
    dz[1:-1, :, :] = (grid[2:, :, :] - grid[:-2, :, :]) / (2 * h)
    
    # backward difference (calculate for grid idx [-1])
    dz[-1, :, :] = (grid[-1, :, :] - grid[-2, :, :]) / h

    return dz

def calculate_gradient(grid: torch.tensor):
    dx = calculate_dx(grid)
    dy = calculate_dy(grid)
    dz = calculate_dz(grid)
    
    grad_norm = torch.sqrt(dx**2 + dy**2 + dz**2)
    gradients = torch.stack([dx, dy, dz], dim=-1)
    
    # debug
    print(f"Gradient Shape: {gradients.shape}")
    print(f"Gradient Norm Shape: {grad_norm.shape}")
    
    return gradients, grad_norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dims', type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--raw_data_file_path', type=str, default="/media/data/qadwu/volume/vortices/vorts1.data")
    parser.add_argument('--tfn_file_path', type=str, default="/home/kctung/Projects/instant-vnr-pytorch/bindings/ovr/data/configs/vorts_shadow.json")
    parser.add_argument('--light_dir_file_path', type=str, default="../VAE_Reconstructed_triplane.pt")
    
    args = parser.parse_args()
    
    try:
        with open(args.light_dir_file_path, 'r') as f:
            loaded_file = json.load(f)
            light_dirs = cartesian_to_spherical_coords(np.array(loaded_file['light_dir_cartesian']))
            print(f"loaded light directions (spherical coords): {light_dirs}")
            # normalize the spherical coordinates to 0~1 (to comply with shadow sampler)
            light_dirs[:,0] = (light_dirs[:,0] % (2*np.pi)) / (2*np.pi)
            light_dirs[:,1] = light_dirs[:,1] / np.pi
            print(f"normalized light directions (spherical coords): {light_dirs}")
    except FileNotFoundError:
        print("Error: 'example.json' not found.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in 'example.json'.")
    
    sampler = create_sampler("structuredRegular", "cuda", dims=args.dims, dtype=args.dtype, n_channels=1, filename=args.raw_data_file_path)
    data_res = args.dims
    chunk_size = 65536*192
    
    # just decode the first light dir
    targets = []
    for coord_chunk in generate_coords_chunks(data_res, chunk_size):
        target = torch.empty([coord_chunk.shape[0], 1]).float().cuda()
        print(f"light dir: {light_dirs[0]}")
        decode_shadow(sampler, coord_chunk, target, light_dirs[0], args.tfn_file_path)
        targets.append(target.cpu())
    targets = torch.cat(targets, dim=0)
    targets = targets.reshape([data_res[2], data_res[1], data_res[0]])
    
    gradients, grad_norm = calculate_gradient(targets)
    import pdb; pdb.set_trace()
    