import sys, os
# Add parent directory to sys.path
# TODO: make it more flexible to call timevarying_data_helper anywhere
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from timevarying_data_helper import *
import argparse
from pysampler import create_sampler
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--raw_data_file_path', type=str, default="/media/data/qadwu/volume/vortices/vorts1.data")
    parser.add_argument('--tfn_file_path', type=str, default="/home/kctung/Projects/instant-vnr-pytorch/bindings/ovr/data/configs/vorts_shadow.json")
    # parser.add_argument('--light_dir_file_path', type=str, default="../VAE_Reconstructed_triplane.pt")
    # NOTE: currently use this script to patch pre-sampled points for existing SIREN model
    parser.add_argument('--SIREN_model_path', type=str, default="/home/kctung/Projects/instant-vnr-pytorch/bindings/ovr/data/configs/vorts_shadow.json")
    args = parser.parse_args()

    loaded_model = torch.load(args.SIREN_model_path, map_location="cpu")
    
    light_dirs = loaded_model['light_dir_cartesian']
    
    sampler = create_sampler("structuredRegular", "cuda", dims=args.dims, dtype=args.dtype, n_channels=1, filename=args.raw_data_file_path)
    data_res = args.dims
    
    # create RandomlyGenerateLightDir just to get the pre-sample coords and values
    # no use of high grad points, so give if_gradient false
    dataset = RandomlyGenerateLightDir(sampler=sampler, n_instances=len(light_dirs), 
                                       tfn=args.tfn_file_path, sample_batch_size=None, 
                                       light_dir_cartesian=light_dirs, resolution=data_res, if_gradient=False)
    # instead, directly call get_uniformly_sampled_points to get presampled points
    coord_groups, value_groups = dataset.get_uniformly_sampled_points()
    
    # coord_groups and value_groups are the lists of tensor
    loaded_model['pre_sampled_coord_groups']=coord_groups
    loaded_model['pre_sampled_value_groups']=value_groups
    
    path = Path(args.SIREN_model_path)
    
    torch.save(loaded_model, os.path.join(path.parent, path.stem + "_pre_sampled.pt"))