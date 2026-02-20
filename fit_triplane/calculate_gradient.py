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

def calculate_gradient_chunked(grid: torch.Tensor, z_chunk_size=64):
    """
    Calculate gradient in Z-axis chunks with proper boundary handling.
    
    Strategy:
    - Each chunk overlaps by 1 slice on each side (for central differences)
    - Compute gradient for the full chunk including overlap
    - Extract only the interior result (excluding overlap regions)
    """
    Z, Y, X = grid.shape
    
    dx_chunks = []
    dy_chunks = []
    dz_chunks = []
    
    for z_start in range(0, Z, z_chunk_size):
        z_end = min(z_start + z_chunk_size, Z)
        
        # Determine overlap: expand chunk by 1 on each side for central differences
        z_start_with_overlap = max(0, z_start - 1)
        z_end_with_overlap = min(Z, z_end + 1)
        
        # Extract chunk with overlap
        chunk = grid[z_start_with_overlap:z_end_with_overlap, :, :].cuda()
        
        # Calculate gradients on the overlapped chunk
        dx_chunk = calculate_dx(chunk)
        dy_chunk = calculate_dy(chunk)
        dz_chunk = calculate_dz(chunk)
        
        # Determine which slices to keep (strip overlap)
        # If z_start > 0, we added 1 slice at the beginning, so skip it
        # If z_end < Z, we added 1 slice at the end, so skip it
        keep_start = 1 if z_start > 0 else 0
        keep_end = dz_chunk.shape[0] - (1 if z_end < Z else 0)
        
        # Extract interior result (without overlap regions)
        dx_chunks.append(dx_chunk[keep_start:keep_end].cpu())
        dy_chunks.append(dy_chunk[keep_start:keep_end].cpu())
        dz_chunks.append(dz_chunk[keep_start:keep_end].cpu())
        
        # Free GPU memory
        del chunk, dx_chunk, dy_chunk, dz_chunk
        torch.cuda.empty_cache()
    
    # Concatenate all chunks
    dx = torch.cat(dx_chunks, dim=0)
    dy = torch.cat(dy_chunks, dim=0)
    dz = torch.cat(dz_chunks, dim=0)
    
    grad_norm = torch.sqrt(dx**2 + dy**2 + dz**2)
    gradients = torch.stack([dx, dy, dz], dim=-1)
    
    print(f"Gradient Shape: {gradients.shape}")
    print(f"Gradient Norm Shape: {grad_norm.shape}")
    
    return gradients, grad_norm

def calculate_gradient_norm_chunked(grid: torch.Tensor, z_chunk_size=128):
    """
    Only calculate gradient norm in Z-axis chunks with proper boundary handling.
    No save gradient for more efficient memory usage.
    
    Strategy:
    - Each chunk overlaps by 1 slice on each side (for central differences)
    - Compute gradient for the full chunk including overlap
    - Extract only the interior result (excluding overlap regions)
    """
    Z, Y, X = grid.shape
    
    norm_chunks = []
    
    for z_start in range(0, Z, z_chunk_size):
        z_end = min(z_start + z_chunk_size, Z)
        
        # Determine overlap: expand chunk by 1 on each side for central differences
        z_start_with_overlap = max(0, z_start - 1)
        z_end_with_overlap = min(Z, z_end + 1)
        
        # Extract chunk with overlap
        # move chunk to gpu for faster computation
        chunk = grid[z_start_with_overlap:z_end_with_overlap, :, :].cuda()
        
        # Calculate gradients on the overlapped chunk
        dx_chunk = calculate_dx(chunk)
        dy_chunk = calculate_dy(chunk)
        dz_chunk = calculate_dz(chunk)
        
        # Determine which slices to keep (strip overlap)
        # If z_start > 0, we added 1 slice at the beginning, so skip it
        # If z_end < Z, we added 1 slice at the end, so skip it
        keep_start = 1 if z_start > 0 else 0
        keep_end = dz_chunk.shape[0] - (1 if z_end < Z else 0)
        
        # Extract interior result (without overlap regions)
        # move the results back to cpu for storage
        norm_chunks.append(torch.sqrt(dx_chunk[keep_start:keep_end]**2 + dy_chunk[keep_start:keep_end]**2 + dz_chunk[keep_start:keep_end]**2).cpu())
        
        # Free GPU memory
        del chunk, dx_chunk, dy_chunk, dz_chunk
        torch.cuda.empty_cache()
    
    # Concatenate all chunks
    grad_norm = torch.cat(norm_chunks, dim=0)
    
    print(f"Gradient Norm Shape: {grad_norm.shape}")
    
    return grad_norm

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
    