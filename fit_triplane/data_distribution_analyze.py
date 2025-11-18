from easydict import EasyDict as edict
import argparse
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import os, sys
import json
from fit import Triplane, Network
import matplotlib.pyplot as plt
from pysampler import create_sampler, decode_shadow
from fit_shadow_randomly_generate import MLP_TCNN

# Add parent directory to sys.path
# TODO: make it more flexible to call timevarying_data_helper anywhere
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from timevarying_data_helper import cartesian_to_spherical_coords

# for debug
def only_decode_raw_shadow(sampler, data_res, chunk_size, tfn_file_path, angle=[0.5, 0.5]):
    targets = []
    for coord_chunk in generate_coords_chunks(data_res, chunk_size):
        target = torch.empty([coord_chunk.shape[0], 1]).float().cuda()
        decode_shadow(sampler, coord_chunk, target, angle, tfn_file_path)
        targets.append(target)
    
    targets = torch.cat(targets, dim=0)
    
    # targets.detach().cpu().numpy().astype(np.float32).tofile(f"test_shadow_volume.bin")
    return targets

def generate_coords_chunks(data_res, chunk_size, device='cuda'):
    """Yield chunks of coordinates from the full 3D grid."""
    gridz, gridy, gridx = torch.meshgrid(
        torch.linspace(0, 1, data_res[2]),  # z slowest
        torch.linspace(0, 1, data_res[1]),
        torch.linspace(0, 1, data_res[0]),  # x fastest
        indexing='ij'
    )
    # the accessing pattern in flattened volume: [1,0,0], [2,0,0], [3,0,0] ... (x change fastest)
    coords = torch.stack([gridx, gridy, gridz], dim=3).reshape(-1, 3)  # [N, 3]
    
    for start in range(0, coords.shape[0], chunk_size):
        end = start + chunk_size
        # allocate memory on CPU, only move to GPU when used for model inference
        yield coords[start:end].to(device)

def inference(n_instances, data_res, chunk_size, value_range, triplane, net, light_dirs, tfn_file_path, loaded_model, loaded_model_2=None):
    psnr_list = []
    with torch.no_grad():
        for batch_idx in range(n_instances):
            net.load_state_dict(loaded_model['net_state_dict'])
            triplane.load_state_dict(loaded_model['triplane_state_dict'])
            preds = []
            for coord_chunk in generate_coords_chunks(data_res, chunk_size):
                preds.append(net(triplane[batch_idx](coord_chunk, 0)).cpu())
            # outputs = net(triplane[batch_idx](coords, 0))
            # outputs = outputs.view(raw_data.shape)
            outputs = torch.cat(preds, dim=0)
            print("finish prediction 1")
            if loaded_model_2:
                net.load_state_dict(loaded_model_2['net_state_dict'])
                triplane.load_state_dict(loaded_model_2['triplane_state_dict'])
                preds = []
                for coord_chunk in generate_coords_chunks(data_res, chunk_size):
                    preds.append(net(triplane[batch_idx](coord_chunk, 0)).cpu())
                # outputs = net(triplane[batch_idx](coords, 0))
                # outputs = outputs.view(raw_data.shape)
                outputs_2 = torch.cat(preds, dim=0)
            print("finish prediction 2")
            targets = []
            for coord_chunk in generate_coords_chunks(data_res, chunk_size):
                target = torch.empty([coord_chunk.shape[0], 1]).float().cuda()
                # TODO: need to convert light dir from radiance to normalized value (0~1)
                decode_shadow(sampler, coord_chunk, target, light_dirs[batch_idx], tfn_file_path)
                targets.append(target.cpu())
            targets = torch.cat(targets, dim=0)
            print("finish decoding")
            # Plot histogram
            plt.figure(figsize=(6,4))
            plt.hist(outputs.numpy(), bins=100, color='blue', alpha=0.8, log=True)
            if loaded_model_2:
                plt.hist(outputs_2.numpy(), bins=100, color='green', alpha=0.8, log=True)
            plt.hist(targets.numpy(), bins=100, color='red', alpha=0.8, log=True)
            plt.title("Value Distribution of Shadow Coefficient Volume")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig("value_distribution_pred.png")
            loss = F.mse_loss(outputs, targets)
            PSNR = (20 * torch.log10(value_range / torch.sqrt(loss))).cpu()
            print("idx:", batch_idx, " psnr:", PSNR)
            psnr_list.append(PSNR)
            if loaded_model_2:
                loss = F.mse_loss(outputs_2, targets)
                PSNR = (20 * torch.log10(value_range / torch.sqrt(loss))).cpu()
                print("idx", batch_idx, " psnr of second model:", PSNR)
            import pdb; pdb.set_trace()
            # ssim_list.append(structural_similarity_index_measure(outputs, raw_data, data_range=1.0).item())
            # save the GPU memory 
            del outputs, targets, loss
            torch.cuda.empty_cache()
    return psnr_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_timevarying.json')
    parser.add_argument('--triplane_file_path', type=str, default="../VAE_Reconstructed_triplane.pt")
    parser.add_argument('--triplane_file_path_2', type=str, default=None)
    parser.add_argument('--dims', type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--raw_data_file_path', type=str, default="/media/data/qadwu/volume/vortices/vorts1.data")
    parser.add_argument('--tfn_file_path', type=str, default="/home/kctung/Projects/instant-vnr-pytorch/bindings/ovr/data/configs/vorts_shadow.json")
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    config = edict(config)
    
    # data_res = [128, 128, 128]
    data_res = args.dims
    chunk_size = 65536*192

    loaded_model = torch.load(args.triplane_file_path)
    n_instances = len(loaded_model['light_dir_cartesian'])
    
    if args.triplane_file_path_2:
        loaded_model_2 = torch.load(args.triplane_file_path_2)
    else:
        loaded_model_2 = None

    # net = Network(
    #     d_in=config.channel,
    #     d_hid=config.n_hid,
    #     n_layers=config.n_layers,
    #     d_out=config.n_labels,
    #     init_type="geo_init",
    # ).cuda()
    
    net = MLP_TCNN(n_input_dims=config.channel, n_output_dims=config.n_labels,
            n_hidden_layers=config.n_layers, n_neurons=config.n_hid,
            activation="ReLU", output_activation="None")

    # instantiate multiple triplanes (each instance has its own triplane)
    triplane = [Triplane(
        reso=config.resolution,
        channel=config.channel,
        init_type="geo_init",
        objname=None,
    ).cuda() for _ in range(n_instances)]
    triplane = nn.ModuleList(triplane)
    
    # volume reconstructed by triplane should between 0~1
    value_range = 1.0
    
    sampler = create_sampler("structuredRegular", "cuda", dims=args.dims, dtype=args.dtype, n_channels=1, filename=args.raw_data_file_path)
    
    light_dirs = cartesian_to_spherical_coords(np.array(loaded_model['light_dir_cartesian']))
    light_dirs_2 = cartesian_to_spherical_coords(np.array(loaded_model_2['light_dir_cartesian'])) if loaded_model_2 else None
    print(f"light dirs 1: {light_dirs}")
    print(f"light dirs 2: {light_dirs_2}")
    
    # normalize the spherical coordinates to 0~1 (to comply with shadow sampler)
    light_dirs[:,0] = (light_dirs[:,0] % (2*np.pi)) / (2*np.pi)
    light_dirs[:,1] = light_dirs[:,1] / np.pi
    psnr_list = inference(n_instances, data_res, chunk_size, value_range, triplane, net, light_dirs, args.tfn_file_path, loaded_model, loaded_model_2)
    
    print(psnr_list)
    