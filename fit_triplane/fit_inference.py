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
from visualize_triplane import plot_single_channel

# Add parent directory to sys.path
# TODO: make it more flexible to call timevarying_data_helper anywhere
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from timevarying_data_helper import TimevaryingDataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_timevarying.json')
    parser.add_argument('--triplane_file_path', type=str, default="../logs/UNet_diffusion_exp/20250624-162156/Diffusion_VAE_Reconstructed_triplane.pt")
    parser.add_argument('--timesteps_to_store', type=int, default=50)
    parser.add_argument('--result_plot_path', type=str, default="psnr_plot.png")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    config = edict(config)
    # assert len(config.fixmlp) > 0

    net = Network(
        d_in=config.channel,
        d_hid=config.n_hid,
        n_layers=config.n_layers,
        d_out=config.n_labels,
        init_type="geo_init",
    ).cuda()

    n_timesteps = 90
    data_res = [128, 128, 128]
    # instantiate multiple triplanes (each timestep has its own triplane)
    triplane = [Triplane(
        reso=config.resolution,
        channel=config.channel,
        init_type="geo_init",
        objname=None,
    ).cuda() for _ in range(n_timesteps)]
    triplane = nn.ModuleList(triplane)

    # prepare raw volumes dataset for evaluation
    train_dataloader = torch.utils.data.DataLoader(
        TimevaryingDataset(
            raw_data_prefix="/home/kctung/vortices",
            raw_data_filename_without_timestep="vorts",
            file_ext="data",
            res=data_res,
            n_timesteps=n_timesteps,
            n_channels=1,
        ),
        batch_size=1,
        shuffle=False)
    # TODO: value range should be retrieve from SampleTimevaryingDataset class (which is more reasonable)
    value_range = 1.0

    # loaded_model = torch.load("ch_32_saved_model.ckpt")
    # loaded_model = torch.load("../VAE_Reconstructed_triplane.pt")
    # loaded_model = torch.load("../Diffusion_Reconstructed_triplane.pt")
    # loaded_model = torch.load("../VAE_Reconstructed_triplane_ch_32.pt")
    # loaded_model = torch.load("../logs/triplane_AE_model_a/20250619-000758/Diffusion_VAE_Reconstructed_triplane.pt")
    loaded_model = torch.load(args.triplane_file_path)
    net.load_state_dict(loaded_model['net_state_dict'])
    triplane.load_state_dict(loaded_model['triplane_state_dict'])

    # generate grid for reconstruction
    with torch.no_grad():
        gridz, gridy, gridx = torch.meshgrid(
            torch.linspace(0, 1, data_res[2]), # z-coords as slowest-changing (outermost) coords
            torch.linspace(0, 1, data_res[1]),
            torch.linspace(0, 1, data_res[0]), # x-coords as fastest-changing (innermost) coords
            indexing='ij'
        )
        # the accessing pattern in flattened volume: [1,0,0], [2,0,0], [3,0,0] ... (x change fastest)
        coords = torch.stack([gridx, gridy, gridz], dim=3)
        coords = coords.reshape([-1, 3])
        coords = coords.cuda()
    
    # from torchmetrics.functional.image import structural_similarity_index_measure
    
    psnr_list = []
    ssim_list = []
    
    with torch.no_grad():
        for batch_idx, raw_data in enumerate(train_dataloader):
            outputs = net(triplane[batch_idx](coords, 0))
            outputs = outputs.view(raw_data.shape)
            loss = F.mse_loss(outputs, raw_data)
            psnr_list.append((20 * torch.log10(value_range / torch.sqrt(loss))).item())
            # import pdb; pdb.set_trace()
            # ssim_list.append(structural_similarity_index_measure(outputs, raw_data, data_range=1.0).item())
            if batch_idx == args.timesteps_to_store:
                outputs.detach().cpu().numpy().astype(np.float32).tofile(f"pred_at_timestep_{batch_idx}.bin")
    
    for i in range(len(psnr_list)):
        print(f"timestep {i} - PSNR: {psnr_list[i]}")
        # print(psnr_list[i], ssim_list[i])
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(psnr_list)), psnr_list, label='PSNR')
    plt.xlabel('Timestep')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR across Timesteps')
    plt.grid(True)
    plt.legend()
    plt.savefig(args.result_plot_path)
    plt.close()