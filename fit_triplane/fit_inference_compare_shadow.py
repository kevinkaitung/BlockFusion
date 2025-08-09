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

# Add parent directory to sys.path
# TODO: make it more flexible to call timevarying_data_helper anywhere
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from timevarying_data_helper import TimevaryingDataset, ShadowVolumesDataset

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
        yield coords[start:end].to(device)

def inference(train_dataloader, data_res, chunk_size, value_range, triplane, net, timestep_to_store, filename_prefix):
    psnr_list = []
    ssim_list = []
    
    # from torchmetrics.functional.image import structural_similarity_index_measure
    with torch.no_grad():
        for batch_idx, raw_data in enumerate(train_dataloader):
            preds = []
            for coord_chunk in generate_coords_chunks(data_res, chunk_size):
                preds.append(net(triplane[batch_idx](coord_chunk, 0)))
            # outputs = net(triplane[batch_idx](coords, 0))
            # outputs = outputs.view(raw_data.shape)
            outputs = torch.cat(preds, dim=0).view(raw_data.shape)
            loss = F.mse_loss(outputs, raw_data)
            psnr_list.append((20 * torch.log10(value_range / torch.sqrt(loss))).cpu())
            # import pdb; pdb.set_trace()
            # ssim_list.append(structural_similarity_index_measure(outputs, raw_data, data_range=1.0).item())
            if batch_idx == timestep_to_store:
                outputs.detach().cpu().numpy().astype(np.float32).tofile(f"{filename_prefix}_recons_at_timestep_{batch_idx}.bin")
                raw_data.detach().cpu().numpy().astype(np.float32).tofile(f"raw_volume_at_timestep_{batch_idx}.bin")
    return psnr_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_timevarying.json')
    parser.add_argument('--timesteps_to_store', type=int, default=50)
    parser.add_argument('--result_plot_name', type=str, default="psnr_plot")
    parser.add_argument('--triplane_file_paths', type=str, nargs='+', default="../VAE_Reconstructed_triplane.pt")
    # pre-trained triplane model path: "ch_32_saved_model.ckpt"
    parser.add_argument('--triplane_recon_types', type=str, nargs='+', default='vae_recon')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    config = edict(config)
    # assert len(config.fixmlp) > 0
    
    n_instances = 150
    # data_res = [128, 128, 128]
    data_res = [256, 256, 256]
    chunk_size = 65536

    net = Network(
        d_in=config.channel,
        d_hid=config.n_hid,
        n_layers=config.n_layers,
        d_out=config.n_labels,
        init_type="geo_init",
    ).cuda()

    # instantiate multiple triplanes (each timestep has its own triplane)
    triplane = [Triplane(
        reso=config.resolution,
        channel=config.channel,
        init_type="geo_init",
        objname=None,
    ).cuda() for _ in range(n_instances)]
    triplane = nn.ModuleList(triplane)

    # prepare raw volumes dataset for evaluation
    train_dataloader = torch.utils.data.DataLoader(
        ShadowVolumesDataset(
            raw_data_dir="/home/kctung/Ring1Light",
            raw_data_filename_prefix="shadow",
            file_ext="bin",
            res=data_res,
            n_instances=n_instances,
            n_channels=1
        ),
        batch_size=1,
        shuffle=False)
    # TODO: value range should be retrieve from TimevaryingDataset class (which is more reasonable)
    value_range = train_dataloader.dataset.value_range

    # loaded_model = torch.load("ch_32_saved_model.ckpt")
    # loaded_model = torch.load("../VAE_Reconstructed_triplane.pt")
    # loaded_model = torch.load("../Diffusion_Reconstructed_triplane.pt")
    # loaded_model_2 = torch.load(args.triplane_file_path_2)
    
    psnr_lists = []
    for triplane_file_path, recon_type in zip(args.triplane_file_paths, args.triplane_recon_types):
        loaded_model = torch.load(triplane_file_path)

        net.load_state_dict(loaded_model['net_state_dict'])
        triplane.load_state_dict(loaded_model['triplane_state_dict'])
        psnr_lists.append(inference(train_dataloader, data_res, chunk_size, value_range, triplane, net, args.timesteps_to_store, recon_type))
        
        # net.load_state_dict(loaded_model_2['net_state_dict'])
        # triplane.load_state_dict(loaded_model_2['triplane_state_dict'])
        # psnr_model_2 = inference(train_dataloader, coords, value_range, triplane, net, args.timesteps_to_store, args.triplane_recon_type_2)
    
    for idx in range(len(psnr_lists[0])):
        print(f"timestep {idx} - ", end="")
        for j in range(len(psnr_lists)):
            print(f"{args.triplane_recon_types[j]} PSNR: {psnr_lists[j][idx]}, ", end="")
        print("")
        # print(psnr_list[i], ssim_list[i])
    
    # After the PSNR printing loop, add:
    plt.figure(figsize=(10, 6))
    for idx in range(len(psnr_lists)):
        plt.plot(range(len(psnr_lists[idx])), psnr_lists[idx], label=f'{args.triplane_recon_types[idx]} PSNR (avg: {torch.mean(torch.stack(psnr_lists[idx])):0,.4f})')
    plt.xlabel('Timestep')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR across Timesteps')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{args.result_plot_name}.png')
    plt.close()