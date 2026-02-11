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
    parser.add_argument('--raw_data_dir', type=str, default="/media/data/qadwu/volume/vortices/vorts1.data")
    # NOTE: currently use this script to patch pre-sampled points for existing SIREN model
    parser.add_argument('--SIREN_model_path', type=str, default="/home/kctung/Projects/instant-vnr-pytorch/bindings/ovr/data/configs/vorts_shadow.json")
    args = parser.parse_args()

    loaded_model = torch.load(args.SIREN_model_path, map_location="cpu")
    
    all_timesteps = loaded_model['timesteps']
    
    data_res = args.dims
    
    # create TimevaryingDataset_with_Sampler just to get the pre-sample coords and values
    dataset = TimevaryingDataset_with_Sampler(
        raw_data_dir=args.raw_data_dir,
        # HACK: just assume volume names start with "timestep_"
        raw_data_filename_without_timestep="timestep_",
        file_ext="bin",
        res=data_res,
        data_type=args.dtype,
        n_instances=len(all_timesteps),
        n_channels=1,
        timesteps=all_timesteps,
        sample_batch_size=None,
        if_get_presampled_points=True
    )
    
    
    coord_groups = dataset.selected_coord_groups
    value_groups = dataset.selected_value_groups
    
    # coord_groups and value_groups are the lists of tensor
    loaded_model['pre_sampled_coord_groups']=coord_groups
    loaded_model['pre_sampled_value_groups']=value_groups
    
    path = Path(args.SIREN_model_path)
    
    torch.save(loaded_model, os.path.join(path.parent, path.stem + "_pre_sampled.pt"))