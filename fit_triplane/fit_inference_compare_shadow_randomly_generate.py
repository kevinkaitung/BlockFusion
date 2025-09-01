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

# Add parent directory to sys.path
# TODO: make it more flexible to call timevarying_data_helper anywhere
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from timevarying_data_helper import ShadowVolumesDataset, RandomlyGenerateLightDir

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

def inference(dataset, data_res, chunk_size, value_range, triplane, net, instances_to_store, filename_prefix):
    psnr_list = []
    ssim_list = []
    # from torchmetrics.functional.image import structural_similarity_index_measure
    with torch.no_grad():
        for batch_idx in range(len(dataset)):
        # hacky way to only evaluate one instance
        # if True:
        #     batch_idx = 100
            preds = []
            targets = []
            # print("before generate coords_chunk: allocated:", torch.cuda.memory.memory_allocated() / 1024**3, " reserved:", torch.cuda.memory.memory_reserved() / 1024**3)
            for coord_chunk in generate_coords_chunks(data_res, chunk_size):
                # print(f"{batch_idx}:before allocated:", torch.cuda.memory.memory_allocated() / 1024**3, " reserved:", torch.cuda.memory.memory_reserved() / 1024**3)
                preds.append(net(triplane[batch_idx](coord_chunk, 0)))
                targets.append(dataset.decode_ith_shadow_volume(batch_idx, coord_chunk)[2])
                # print(f"{batch_idx}:after allocated:", torch.cuda.memory.memory_allocated() / 1024**3, " reserved:", torch.cuda.memory.memory_reserved() / 1024**3)
            # outputs = net(triplane[batch_idx](coords, 0))
            # outputs = outputs.view(raw_data.shape)
            outputs = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            loss = F.mse_loss(outputs, targets)
            PSNR = (20 * torch.log10(value_range / torch.sqrt(loss))).cpu()
            psnr_list.append(PSNR)
            print("idx:", batch_idx, " psnr:", PSNR)
            # ssim_list.append(structural_similarity_index_measure(outputs, raw_data, data_range=1.0).item())
            # print("after loss calculation: allocated:", torch.cuda.memory.memory_allocated() / 1024**3, " reserved:", torch.cuda.memory.memory_reserved() / 1024**3)
            if batch_idx in instances_to_store:
                outputs.detach().cpu().numpy().astype(np.float32).tofile(f"{filename_prefix}_recons_at_instance_{batch_idx}.bin")
                targets.detach().cpu().numpy().astype(np.float32).tofile(f"raw_volume_at_instance_{batch_idx}.bin")
            # save the GPU memory 
            del outputs, targets, loss
            torch.cuda.empty_cache()
            # print("after deleting: allocated:", torch.cuda.memory.memory_allocated() / 1024**3, " reserved:", torch.cuda.memory.memory_reserved() / 1024**3)
    return psnr_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_timevarying.json')
    parser.add_argument('--instances_to_store', type=int, nargs='+', default=[0])
    parser.add_argument('--result_plot_name', type=str, default="psnr_plot")
    parser.add_argument('--triplane_file_paths', type=str, nargs='+', default="../VAE_Reconstructed_triplane.pt")
    # pre-trained triplane model path: "ch_32_saved_model.ckpt"
    parser.add_argument('--triplane_recon_types', type=str, nargs='+', default='vae_recon')
    
    parser.add_argument('--dims', type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--raw_data_file_path', type=str, default="/media/data/qadwu/volume/vortices/vorts1.data")
    parser.add_argument('--tfn_file_path', type=str, default="/home/kctung/Projects/instant-vnr-pytorch/bindings/ovr/data/configs/vorts_shadow.json")
    parser.add_argument('--n_instances', type=int, default=150, help="Number of shadow volumes to generate")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    config = edict(config)
    # assert len(config.fixmlp) > 0
    
    n_instances = args.n_instances
    # data_res = [128, 128, 128]
    data_res = args.dims
    chunk_size = 65536*192

    net = Network(
        d_in=config.channel,
        d_hid=config.n_hid,
        n_layers=config.n_layers,
        d_out=config.n_labels,
        init_type="geo_init",
    ).cuda()

    # instantiate multiple triplanes (each instance has its own triplane)
    triplane = [Triplane(
        reso=config.resolution,
        channel=config.channel,
        init_type="geo_init",
        objname=None,
    ).cuda() for _ in range(n_instances)]
    triplane = nn.ModuleList(triplane)
    
    sampler = create_sampler("structuredRegular", "cuda", dims=args.dims, dtype=args.dtype, n_channels=1, filename=args.raw_data_file_path)
    # import pdb; pdb.set_trace()
    
    # just use the first triplane to load light directions
    # should compare all sets of triplanes that have the same light direction set
    loaded_model = torch.load(args.triplane_file_paths[0])
    # prepare dataset for evaluation
    dataset = RandomlyGenerateLightDir(
        sampler=sampler,
        n_instances=args.n_instances,
        tfn=args.tfn_file_path,
        sample_batch_size=config.sample_batch_size,
        light_dir_spherical=loaded_model['light_dir_spherical']
    )
    value_range = dataset.value_range
    
    psnr_lists = []
    for triplane_file_path, recon_type in zip(args.triplane_file_paths, args.triplane_recon_types):
        loaded_model = torch.load(triplane_file_path)

        net.load_state_dict(loaded_model['net_state_dict'])
        triplane.load_state_dict(loaded_model['triplane_state_dict'])
        psnr_lists.append(inference(dataset, data_res, chunk_size, value_range, triplane, net, args.instances_to_store, recon_type))
    
    for idx in range(len(psnr_lists[0])):
        print(f"instance {idx} - ", end="")
        for j in range(len(psnr_lists)):
            print(f"{args.triplane_recon_types[j]} PSNR: {psnr_lists[j][idx]}, ", end="")
        print("")
        # print(psnr_list[i], ssim_list[i])
    
    # After the PSNR printing loop, add:
    plt.figure(figsize=(10, 6))
    for idx in range(len(psnr_lists)):
        plt.plot(range(len(psnr_lists[idx])), psnr_lists[idx], label=f'{args.triplane_recon_types[idx]} PSNR (avg: {torch.mean(torch.stack(psnr_lists[idx])):0,.4f})')
    plt.xlabel('Instance')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR across Instances')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{args.result_plot_name}.png')
    plt.close()