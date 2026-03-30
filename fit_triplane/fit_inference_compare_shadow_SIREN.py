from pysampler import create_sampler, decode_shadow
from easydict import EasyDict as edict
import argparse
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import os, sys
import json
from fit import Triplane, Network
from fit_shadow_randomly_generate import MLP_TCNN
import matplotlib.pyplot as plt
from tqdm import tqdm
from networks import NeurCompNet

# Add parent directory to sys.path
# TODO: make it more flexible to call timevarying_data_helper anywhere
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from timevarying_data_helper import ShadowVolumesDataset, RandomlyGenerateLightDir, cartesian_to_spherical_coords, spherical_to_cartesian_coords

# for debug
def only_decode_raw_shadow(sampler, data_res, chunk_size, tfn_file_path, angle=[0.5, 0.5]):
    targets = []
    for coord_chunk in generate_grid_coords_chunks(data_res, chunk_size):
        target = torch.empty([coord_chunk.shape[0], 1]).float().cuda()
        decode_shadow(sampler, coord_chunk, target, angle, tfn_file_path)
        targets.append(target)
    
    targets = torch.cat(targets, dim=0)
    
    # targets.detach().cpu().numpy().astype(np.float32).tofile(f"test_shadow_volume.bin")
    return targets
        
def generate_grid_coords_chunks(data_res, chunk_size, device='cuda'):
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

def generate_random_coords_chunks(total_batch_size, chunk_size, device='cuda'):
    """Yield chunks of randomly generated coordinates."""
    # the accessing pattern in flattened volume: [1,0,0], [2,0,0], [3,0,0] ... (x change fastest)
    coords = torch.rand([total_batch_size, 3], dtype=torch.float32, device="cuda")  # [N, 3]
    
    for start in range(0, coords.shape[0], chunk_size):
        end = start + chunk_size
        # allocate memory on CPU, only move to GPU when used for model inference
        yield coords[start:end].to(device)

def inference(dataset, data_res, chunk_size, value_range, nets, instances_to_store, filename_prefix, sampled_batch_size=5000000):
    psnr_list = []
    psnr_fast_list = []
    ssim_list = []
    # from torchmetrics.functional.image import structural_similarity_index_measure
    with torch.no_grad():
        ### section to evaluate the whole volume
        # for batch_idx in tqdm(range(len(dataset))):
        # # hacky way to only evaluate one instance
        # # if True:
        # #     batch_idx = 100
        #     # preds = []
        #     # targets = []
        #     # print("before generate coords_chunk: allocated:", torch.cuda.memory.memory_allocated() / 1024**3, " reserved:", torch.cuda.memory.memory_reserved() / 1024**3)
        #     total_sq_error = 0.0
        #     total_count = 0
        #     this_batch_triplane = triplane[batch_idx].cuda()
        #     for coord_chunk in generate_grid_coords_chunks(data_res, chunk_size):
        #         # print(f"{batch_idx}:before allocated:", torch.cuda.memory.memory_allocated() / 1024**3, " reserved:", torch.cuda.memory.memory_reserved() / 1024**3)
        #         # preds.append(net(this_batch_triplane(coord_chunk, 0)))
        #         # targets.append(dataset.decode_ith_shadow_volume(batch_idx, coord_chunk)[2])
        #         preds = net(this_batch_triplane(coord_chunk, 0))
        #         targets = dataset.decode_ith_shadow_volume(batch_idx, coord_chunk)[2]
        #         # sum of squared errors for this chunk
        #         sq_error = F.mse_loss(preds, targets, reduction="sum")
        #         total_sq_error += sq_error.item()
        #         total_count += targets.numel()
                
        #         # print(f"{batch_idx}:after allocated:", torch.cuda.memory.memory_allocated() / 1024**3, " reserved:", torch.cuda.memory.memory_reserved() / 1024**3)
        #     # outputs = net(triplane[batch_idx](coords, 0))
        #     # outputs = outputs.view(raw_data.shape)
        #     loss = total_sq_error / total_count
            
        #     this_batch_triplane = this_batch_triplane.cpu()
        #     # outputs = torch.cat(preds, dim=0)
        #     # targets = torch.cat(targets, dim=0)
        #     # loss = F.mse_loss(outputs, targets)
        #     # PSNR = (20 * torch.log10(value_range / torch.sqrt(loss))).cpu()
        #     PSNR = (20 * np.log10(value_range / np.sqrt(loss)))
        #     psnr_list.append(PSNR)
        #     print(f"idx:{batch_idx} PSNR eval on the whole volume:{PSNR}")
        #     # ssim_list.append(structural_similarity_index_measure(outputs, raw_data, data_range=1.0).item())
        #     # print("after loss calculation: allocated:", torch.cuda.memory.memory_allocated() / 1024**3, " reserved:", torch.cuda.memory.memory_reserved() / 1024**3)
        #     # if batch_idx in instances_to_store:
        #     #     outputs.detach().cpu().numpy().astype(np.float32).tofile(f"{filename_prefix}_recons_at_instance_{batch_idx}.bin")
        #     #     targets.detach().cpu().numpy().astype(np.float32).tofile(f"raw_volume_at_instance_{batch_idx}.bin")
        #     # save the GPU memory 
        #     # del outputs, targets, loss
        #     # torch.cuda.empty_cache()
        #     # print("after deleting: allocated:", torch.cuda.memory.memory_allocated() / 1024**3, " reserved:", torch.cuda.memory.memory_reserved() / 1024**3)
        ### section end
        ### section to only evaluate on sampled coords
        for batch_idx in tqdm(range(len(dataset))):
            total_sq_error = 0.0
            total_count = 0
            this_batch_net = nets[batch_idx].cuda()
            for coord_chunk in generate_random_coords_chunks(sampled_batch_size, chunk_size):
                with torch.no_grad():
                    preds = this_batch_net(coord_chunk)
                targets = dataset.decode_ith_shadow_volume(batch_idx, coord_chunk)[2]
                # sum of squared errors for this chunk
                sq_error = F.mse_loss(preds, targets, reduction="sum")
                total_sq_error += sq_error.item()
                total_count += targets.numel()
        
            loss = total_sq_error / total_count
            
            this_batch_net = this_batch_net.cpu()
            PSNR = (20 * np.log10(value_range / np.sqrt(loss)))
            psnr_fast_list.append(PSNR)
            print(f"idx:{batch_idx} PSNR eval only on sampled coords:{PSNR}")
        ### section end
    # return psnr_list, psnr_fast_list
    return psnr_fast_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, nargs='+', default='base_timevarying.json')
    parser.add_argument('--instances_to_store', type=int, nargs='+', default=[0])
    parser.add_argument('--result_plot_name', type=str, default="psnr_plot")
    parser.add_argument('--SIREN_file_paths', type=str, nargs='+', default="")
    # pre-trained triplane model path: "ch_32_saved_model.ckpt"
    parser.add_argument('--SIREN_recon_types', type=str, nargs='+', default='SIREN_diffusion')
    
    parser.add_argument('--dims', type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--raw_data_file_path', type=str, default="/media/data/qadwu/volume/vortices/vorts1.data")
    parser.add_argument('--tfn_file_path', type=str, default="/home/kctung/Projects/instant-vnr-pytorch/bindings/ovr/data/configs/vorts_shadow.json")
    # parser.add_argument('--n_instances', type=int, default=150, help="Number of shadow volumes to generate")
    args = parser.parse_args()
    
    # n_instances = args.n_instances
    # data_res = [128, 128, 128]
    data_res = args.dims
    chunk_size = 65536*192
    
    sampler = create_sampler("structuredRegular", "cuda", dims=args.dims, dtype=args.dtype, n_channels=1, filename=args.raw_data_file_path)
    # import pdb; pdb.set_trace()
    
    # just use the first triplane to load light directions
    # should compare all sets of triplanes that have the same light direction set
    # loaded_model = torch.load(args.triplane_file_paths[0])
    # # TODO: think of how to compare
    # assert args.n_instances == len(loaded_model['light_dir_cartesian'])
    
    # ### section of hacky way to temporarily bypass the problem (not used anymore)
    # spherical_coords = cartesian_to_spherical_coords(np.array(loaded_model['light_dir_cartesian']))
    # print(f"{1}: {spherical_coords.shape}")
    # theta=spherical_coords[:,0].astype(float)
    # phi=spherical_coords[:,1].astype(float)
    # print(f"{2}: {theta.shape}, {phi.shape}")
    
    # theta=theta*np.pi
    # phi=phi*2.0*np.pi
    # print(f"{3}: {np.stack([theta, phi], axis=-1).shape}")
    # temp = spherical_to_cartesian_coords(np.stack([theta, phi], axis=-1))
    # print(f"{4}: {temp.shape}")
    # temp1 = cartesian_to_spherical_coords(temp)
    # print(f"{5}: {temp1.shape}")
    # corrected_spherical_coords = np.stack([(temp1[:,0] % 2*np.pi)/(2*np.pi), temp1[:,1]/np.pi], axis=-1)
    # print(f"{6}: {corrected_spherical_coords.shape}")
    # ### section end
    
    psnr_lists = []
    psnr_fast_lists = []
    light_dir_lists = []
    for config_file_path, SIREN_file_path, recon_type in zip(args.configs, args.SIREN_file_paths, args.SIREN_recon_types):
        
        loaded_model = torch.load(SIREN_file_path, map_location="cpu")
        
        n_instances = len(loaded_model['light_dir_cartesian'])
        
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        config = edict(config)
        # assert len(config.fixmlp) > 0
        
        # prepare dataset for evaluation
        dataset = RandomlyGenerateLightDir(
            sampler=sampler,
            n_instances=n_instances,
            tfn=args.tfn_file_path,
            sample_batch_size=config.sample_batch_size,
            light_dir_spherical=cartesian_to_spherical_coords(np.array(loaded_model['light_dir_cartesian'])).tolist()
            # light_dir_spherical=loaded_model['light_dir_spherical']
            # light_dir_spherical=corrected_spherical_coords.tolist()
        )
        value_range = dataset.value_range
        
        nets = [
            NeurCompNet(n_input_dims=3, n_output_dims=config.n_labels, bias=False, n_hidden_layers=config.n_layers, n_neurons=config.n_hid, is_residual=True)
            for _ in range(n_instances)]
        nets = nn.ModuleList(nets)

        nets.load_state_dict(loaded_model['net_state_dict'])
        
        # psnr_list, psnr_fast_list = inference(dataset, data_res, chunk_size, value_range, triplane, net, args.instances_to_store, recon_type)
        psnr_list = inference(dataset, data_res, chunk_size, value_range, nets, args.instances_to_store, recon_type)
        psnr_lists.append(psnr_list)
        light_dir_lists.append(loaded_model['light_dir_cartesian'])
        # psnr_fast_lists.append(psnr_fast_list)
    
    # refactor for various #instances
    max_instances = max(len(lst) for lst in psnr_lists)
    
    for idx in range(max_instances):
        print(f"instance {idx} - ", end="")
        for j in range(len(psnr_lists)):
            if idx < len(psnr_lists[j]):  # check if this list has enough elements
                print(f"{args.SIREN_recon_types[j]} PSNR: {psnr_lists[j][idx]}, ", end="")
            else:
                print(f"{args.SIREN_recon_types[j]} PSNR: N / A, ", end="")
        print("")
        # print(psnr_list[i], ssim_list[i])
    
    # After the PSNR printing loop, add:
    plt.figure(figsize=(10, 6))
    for idx in range(len(psnr_lists)):
        plt.plot(range(len(psnr_lists[idx])), psnr_lists[idx], label=f'{args.SIREN_recon_types[idx]} PSNR (avg: {np.mean(np.stack(psnr_lists[idx])):0,.4f})')
        print(f'{args.SIREN_recon_types[idx]} PSNR (avg: {np.mean(np.stack(psnr_lists[idx])):0,.4f})')
        # plt.plot(range(len(psnr_fast_lists[idx])), psnr_fast_lists[idx], label=f'{args.triplane_recon_types[idx]} fast PSNR (avg: {np.mean(np.stack(psnr_fast_lists[idx])):0,.4f})')
    plt.xlabel('Instance')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR across Instances')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{args.result_plot_name}.png')
    plt.close()
    
    info_json = dict()
    for idx in range(len(psnr_lists)):
        info_json[f'{args.SIREN_recon_types[idx]}'] = dict()
        info_json[f'{args.SIREN_recon_types[idx]}']['light_dir_cartesian'] = light_dir_lists[idx]
        info_json[f'{args.SIREN_recon_types[idx]}']['PSNR_recon_quality'] = psnr_lists[idx]
    with open(f"{args.result_plot_name}.json", "w") as f:
        json.dump(info_json, f, indent=4)
    
    # import math
    # # for plot sampled lighting directions
    # group_size = 9
    # plt.figure(figsize=(10, 6))
    # for idx in range(len(psnr_lists)):
    #     num_groups = math.ceil(len(psnr_lists[idx]) / group_size)
    #     for group_idx in range(num_groups):
    #         start = group_idx * group_size
    #         end = (group_idx + 1) * group_size
    #         plt.subplot(math.ceil(num_groups / 2), 2, group_idx+1)
    #         plt.plot(range(len(psnr_lists[idx][start:end])), psnr_lists[idx][start:end], label=f'{args.triplane_recon_types[idx]} PSNR (avg: {np.mean(np.stack(psnr_lists[idx][start:end])):0,.4f})')
    #         plt.xlabel('Instance Idx (center pt at idx 4)')
    #         plt.ylabel('PSNR (dB)')
    #         plt.title(f"Group {group_idx}")
    #         plt.grid(True)
    #         plt.legend()
    # plt.tight_layout()
    # plt.savefig(f'{args.result_plot_name}.png')
    # plt.close()