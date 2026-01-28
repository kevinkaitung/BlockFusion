# pysampler needs to be imported before create_sampler
from pysampler import create_sampler
from tqdm import tqdm
from easydict import EasyDict as edict
import argparse
import logging
from datetime import datetime
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
# import open3d as o3d
# import mcubes, trimesh
import torch
import numpy as np
import os, sys
import json

if torch.cuda.is_available():
    import tinycudann as tcnn

from torch.utils.checkpoint import checkpoint
from fit_shadow_randomly_generate import Triplane, MLP_TCNN, Network

# Add parent directory to sys.path
# TODO: make it more flexible to call timevarying_data_helper anywhere
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from visualize_triplane import plot_single_channel
from timevarying_data_helper import SampleShadowVolumesDataset, RandomlyGenerateLightDir, fibonacci_sphere, cartesian_to_spherical_coords

check_plane_idx = 40
vis_triplane_freq = 2000

def create_optimizer(net, triplane, config, optimizer_type):
    params_to_train = []
    if triplane is not None:
        params_to_train += [{'name':'tri', 'params':triplane.parameters(), 'lr':config.lr_tri}]
    if net is not None:
        params_to_train += [{'name':'net', 'params':net.parameters(), 'lr':config.lr_net}]
    if optimizer_type == "Adam":
        return torch.optim.Adam(params_to_train)
    elif optimizer_type == "SGD":
        return torch.optim.SGD(params_to_train)
    else:
        raise RuntimeError(f"{optimizer_type} optimizer not supported!")

def update_lr(optimizer, epoch, config):
    # TODO: make sure the lr for finetuning is reasonable
    learning_factor = (np.cos(np.pi * epoch / config.max_iters) + 1.0) * 0.5 * (1 - 0.001) + 0.001
    for param_group in optimizer.param_groups:
        if "net" in param_group['name']:
            param_group['lr'] = config.lr_net * learning_factor
        if "tri" in param_group['name']:
            param_group['lr'] = config.lr_tri * learning_factor

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_timevarying.json')
    parser.add_argument("--expname", type=str, default="finetune_VAE_recon_triplanes", help="Experiment name")
    parser.add_argument("--description", type=str, default="", help="Description to experiment")
    # parser.add_argument('--resume_training_model', type=str, default=None)
    # parser.add_argument('--only_finetune_mlp', action='store_true')
    parser.add_argument('--optimizer_type', type=str, default="Adam")
    parser.add_argument('--use_native_mlp', action='store_true')
    
    parser.add_argument('--dims', type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--raw_data_file_path', type=str, default="/media/data/qadwu/volume/vortices/vorts1.data")
    parser.add_argument('--tfn_file_path', type=str, default="/home/kctung/Projects/instant-vnr-pytorch/bindings/ovr/data/configs/vorts_shadow.json")
    parser.add_argument('--n_instances', type=int, default=150, help="Number of shadow volumes to generate")
    parser.add_argument('--selected_light_dirs_file_path', type=str)
    parser.add_argument('--output_activation', type=str, default="None")
    args = parser.parse_args()

    # create directory for saving logs
    base_dir = "../logs"
    os.makedirs(base_dir, exist_ok=True)
    expname_dir = os.path.join(base_dir, args.expname)
    os.makedirs(expname_dir, exist_ok=True)
    run_dir = os.path.join(expname_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    logging_file_md = 'w'
    
    # create tensorboard logger
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_writer = SummaryWriter(log_dir=run_dir)
    
    # prepare python logger
    logging.basicConfig(filename=os.path.join(run_dir, "console_log.log"),
                    format='%(asctime)s %(message)s',
                    filemode=logging_file_md)
    console_logger = logging.getLogger()
    console_logger.setLevel(logging.DEBUG)

    # to suppress matplotlib logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    console_logger.debug("Experiment description: " + args.description)
    # if args.resume_training_model:
    #     console_logger.debug("Resume Training Model Path: " + args.resume_training_model)
    console_logger.debug("Config File Name: " + args.config)
    # console_logger.debug("Only finetune mlp: " + str(args.only_finetune_mlp))
    console_logger.debug("Optimizer Type: " + args.optimizer_type)
    console_logger.debug("Use Native MLP: " + str(args.use_native_mlp))
    console_logger.debug("Path to Raw Data File: " + args.raw_data_file_path)
    console_logger.debug("Path to TFN Data File: " + args.tfn_file_path)
    console_logger.debug("Number of Instances Generated: " + str(args.n_instances))
    if args.selected_light_dirs_file_path:
        console_logger.debug("Selected Light Directions File Path: " + args.selected_light_dirs_file_path)
    console_logger.debug("MLP Output Activation: " + args.output_activation)
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    config = edict(config)

    sampler = create_sampler("structuredRegular", "cuda", dims=args.dims, dtype=args.dtype, n_channels=1, filename=args.raw_data_file_path)
    
    # if selected_light_dirs_file_path is defined, use the light dirs from the file
    if args.selected_light_dirs_file_path:
        with open(args.selected_light_dirs_file_path, 'r') as f:
            selected_light_dirs = json.load(f)
        # only get n_instances light dirs
        all_lighting_dirs_cartesian = np.array(selected_light_dirs["light_dir_cartesian"][:args.n_instances])
        print(f"Check the shape of loaded lighting directions: {all_lighting_dirs_cartesian.shape}")
    # otherwise, generated n_instances points on Fibonacci Sphere
    else:
        # return as np array
        all_lighting_dirs_cartesian = fibonacci_sphere(args.n_instances, False)
    all_lighting_dirs_spherical = cartesian_to_spherical_coords(all_lighting_dirs_cartesian)
    
    # generate permuted indices
    permuted_indices = np.random.permutation(len(all_lighting_dirs_cartesian))
    
    all_lighting_dirs_cartesian = all_lighting_dirs_cartesian[permuted_indices].tolist()
    all_lighting_dirs_spherical = all_lighting_dirs_spherical[permuted_indices].tolist()
    
    
    if args.use_native_mlp:
        net = Network(
            d_in=config.channel,
            d_hid=config.n_hid,
            n_layers=config.n_layers,
            d_out=config.n_labels,
            init_type="geo_init",
        ).cuda()
        # net.load_state_dict(torch.load(config.fixmlp, map_location='cuda'))
    else:
        net = MLP_TCNN(n_input_dims=config.channel, n_output_dims=config.n_labels,
                    n_hidden_layers=config.n_layers, n_neurons=config.n_hid,
                    activation="ReLU", output_activation=args.output_activation)
    
    
    epoch = 0
    # for grouping triplanes and their corresponding optimizers
    assert args.n_instances % config.batch_size == 0, "Number of instances must be divisible by batch size"
    triplane_offsets = []
    current_tri_reso = config.resolution // (2 ** len(config.c2f_scale))
    for idx in range(args.n_instances // config.batch_size):
        start_idx = idx * config.batch_size
        end_idx = min((idx + 1) * config.batch_size, args.n_instances)

        triplane_offsets.append(start_idx)

        # instantiate multiple triplanes (each timestep has its own triplane)
        triplane = [Triplane(
            reso=current_tri_reso,
            channel=config.channel,
            init_type="geo_init",
            objname=None,
        ).cuda() for _ in range(config.batch_size)]
        triplane = nn.ModuleList(triplane)
        
        optimizer = create_optimizer(net, triplane, config, args.optimizer_type)
        
        torch.save({
            'triplane_state_dict': triplane.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'light_dir_spherical': all_lighting_dirs_spherical[start_idx:end_idx],
            'light_dir_cartesian': all_lighting_dirs_cartesian[start_idx:end_idx],
            'offset': start_idx,
            'end_idx': end_idx,
            # 'lr_net': final_lr_net,
            # 'lr_tri': final_lr_tri,
            'triplane_resolution': current_tri_reso,
            'epoch': epoch,
        }, os.path.join(run_dir, f"triplane_offset_{start_idx}.ckpt"))

    # for fine tune model (both for only finetune MLP, finetune triplanes+MLP, and resume training from original triplane overfitting training)
    # if args.resume_training_model:
    #     loaded_model = torch.load(args.resume_training_model)
    #     net.load_state_dict(loaded_model['net_state_dict'])
    #     triplane.load_state_dict(loaded_model['triplane_state_dict'])
    #     # TODO: make sure whether I should use optimizer from pretrained triplane?
    #     # Caution: also need to pass optimizer_type for resuming training from ckpt
    #     if args.only_finetune_mlp:
    #         optimizer = create_optimizer(net, None, config, args.optimizer_type)
    #     else:
    #         optimizer = create_optimizer(net, triplane, config, args.optimizer_type)
    # else:
    
    # if "resume_iter" in config:
    #     start_iter = config.resume_iter
    # else:
    #     start_iter = 0

    num_epoch_to_reload_new_subset = config.subset_reload_iters
    assert config.max_iters % num_epoch_to_reload_new_subset == 0
    num_run = config.max_iters // num_epoch_to_reload_new_subset
    for run in range(num_run):
        
        permuted = np.random.permutation(triplane_offsets).tolist()
        start_epoch = run * num_epoch_to_reload_new_subset
                
        for offset_idx, offset in enumerate(permuted):
            
            subset_model = torch.load(os.path.join(run_dir, f"triplane_offset_{offset}.ckpt"))
            
            # start_epoch = subset_model['epoch']
            current_tri_reso = subset_model['triplane_resolution']
            triplane = [Triplane(
                reso=current_tri_reso,
                channel=config.channel,
                init_type="geo_init",
                objname=None,
            ).cuda() for _ in range(config.batch_size)]
            triplane = nn.ModuleList(triplane)
            triplane.load_state_dict(subset_model['triplane_state_dict'])
            
            # in the last run (upon convergence), freeze MLP and only optimize triplanes
            # if run == num_run - 1:
            if offset_idx > 0:
                # placeholder to create optimizer for complying loaded state dict
                optimizer = create_optimizer(net, triplane, config, args.optimizer_type)
                # still load optimizer state dict first -> for triplanes state
                optimizer.load_state_dict(subset_model['optimizer_state_dict'])
                
                # freeze MLP
                for p in net.parameters():
                    p.requires_grad = False
                
                # remove MLP related optimizer state
                net_idx_to_delete = 0
                for i, param_group in enumerate(optimizer.param_groups):
                    if "net" in param_group['name']:
                        # only if the parameter key appears in the optimizer state
                        # we can access it
                        if param_group['params'][0] in optimizer.state:
                            del optimizer.state[param_group['params'][0]]
                        net_idx_to_delete = i
                        break
                del optimizer.param_groups[net_idx_to_delete]
            else:
                # get optimizer's network parameters
                for param_group in optimizer.param_groups:
                    if "net" in param_group['name']:
                        net_optimizer_param_group = param_group
                        break
                # TODO: make sure to get all state keys of net
                # currently might only support TCNN MLP because it only has one key
                net_optimizer_state_key = net_optimizer_param_group['params'][0]
                if net_optimizer_state_key in optimizer.state:
                    net_optimizer_state = optimizer.state[net_optimizer_state_key]
                else:
                    net_optimizer_state = None
                
                optimizer.load_state_dict(subset_model['optimizer_state_dict'])
                if net_optimizer_state is not None:
                    optimizer.state[net_optimizer_state_key] = net_optimizer_state
                
                for i, param_group in enumerate(optimizer.param_groups):
                    if "net" in param_group['name']:
                        optimizer.param_groups[i] = net_optimizer_param_group
                        break
                # optimizer.param_groups[1] = net_optimizer_param_group
            
            # create DataLoader for this subset
            train_dataloader = torch.utils.data.DataLoader(
            RandomlyGenerateLightDir(
                sampler=sampler,
                n_instances=config.batch_size,
                tfn=args.tfn_file_path,
                sample_batch_size=config.sample_batch_size,
                light_dir_cartesian=subset_model['light_dir_cartesian'],
                resolution=args.dims,
                if_view_transform=True
            ),
            batch_size=config.batch_size,
            shuffle=True)
            value_range = train_dataloader.dataset.value_range
                        
            
            for epoch in tqdm(range(start_epoch + 1, start_epoch + num_epoch_to_reload_new_subset + 1)):
                
                running_loss = 0.0
                
                # loss_list = []

                # for debugging
                if epoch % vis_triplane_freq == 0:
                    for dim in range(3):
                        plot_single_channel(
                            triplane[check_plane_idx].triplane[0][dim][16].detach(), 
                            title=f"plane_offset_{offset}_dim_{dim}_epoch_{epoch}",
                            save_path=os.path.join(run_dir, f"plane_offset_{offset}_dim_{dim}_epoch_{epoch}.png")
                        )

                if epoch in config.c2f_scale:
                    new_reso = int(config.resolution / (2 ** (len(config.c2f_scale) - config.c2f_scale.index(epoch) - 1)))
                    current_tri_reso = new_reso
                    # for debugging
                    for dim in range(3):
                        plot_single_channel(
                            triplane[check_plane_idx].triplane[0][dim][16].detach(), 
                            title=f"plane_offset_{offset}_dim_{dim}_reso_{new_reso}",
                            save_path=os.path.join(run_dir, f"plane_offset_{offset}_dim_{dim}_reso_{new_reso}.png")
                        )
                    for tri in triplane:
                        tri.update_resolution(new_reso)
                    optimizer = create_optimizer(net, triplane, config, args.optimizer_type)
                    update_lr(optimizer, epoch - 1, config)
                    torch.cuda.empty_cache()
                    console_logger.debug(f"Peak memory usage at epoch {epoch}: allocated: {torch.cuda.memory.max_memory_allocated() / 1024**3} reserved: {torch.cuda.memory.max_memory_reserved() / 1024**3}")

                for batch_idx, data in enumerate(train_dataloader):
                    
                    # for debug
                    # start_mem = torch.cuda.memory_allocated()
                    # start_rsv = torch.cuda.memory_reserved()
                    # print(f"start mem: {start_mem / 1024**3:.2f} GB / rsv: {start_rsv / 1024**3:.2f} GB")
                    
                    # data format: tuple(timestep_index, sample_coords, target_values)
                    outputs = []
                    targets = data[2]
                    for timestep, sample_coords in zip(data[0], data[1]):
                        # the output of TCNN MLP would be half type
                        # convert to float type
                        outputs.append(net(triplane[timestep](sample_coords, 0)).float())
                        
                    # outputs[0].shape = [1024, 1]
                    outputs = torch.stack(outputs, dim=0)
                    # outputs.shape = [90, 1024, 1]
                    # targets.shape = [90, 1024, 1]
                    loss = F.mse_loss(outputs, targets)

                    # deleting targets and outputs might not clean up much memory
                    # most of the memory might be consumed by the intermediate results of forward pass
                    # del targets, outputs
                    # torch.cuda.empty_cache()
                    
                    optimizer.zero_grad()
                    # print(f"Before backward Mem Alloc: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB / Max Alloc: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB / Max Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
                    loss.backward()
                    # print(f"Before step Mem Alloc: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB / Max Alloc: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB / Max Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
                    optimizer.step()
                    # print(f"After step Mem Alloc: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB / Max Alloc: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB / Max Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
                    
                    running_loss += loss.item()
                    # del loss
                    # torch.cuda.empty_cache()
                    
                avg_loss = running_loss / len(train_dataloader)
                # loss_list.append(avg_loss)
                PSNR_value = (20 * np.log10(value_range / np.sqrt(avg_loss))).item()
                console_logger.debug(f"Subset {offset}, Epoch {epoch}, Loss: {avg_loss}, Reconstruction PSNR: {(PSNR_value):0,.4f}")
                print(f"Subset {offset}, Epoch {epoch}, Loss: {avg_loss}, , Reconstruction PSNR: {(PSNR_value):0,.4f}")
                tensorboard_writer.add_scalar(f"Loss/Subset_{offset}", avg_loss, epoch)
                tensorboard_writer.add_scalar(f"PSNR/Subset_{offset}", PSNR_value, epoch)
                update_lr(optimizer, epoch, config)
                print(f"Mem Alloc: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB / Max Alloc: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB / Max Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")

            # get the final learning rates of triplane and MLP
            for param_group in optimizer.param_groups:
                if "tri" in param_group['name']:
                    final_lr_tri = param_group['lr']
                else:
                    final_lr_tri = None

            # save this subset training results to disk
            torch.save({
                'net_state_dict': net.state_dict(),
                'triplane_state_dict': triplane.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'light_dir_spherical': subset_model['light_dir_spherical'],
                'light_dir_cartesian': subset_model['light_dir_cartesian'],
                'offset': offset,
                'end_idx': subset_model['end_idx'],
                # 'lr_net': final_lr_net,
                'lr_tri': final_lr_tri,
                'triplane_resolution': current_tri_reso,
                'epoch': epoch,
            }, os.path.join(run_dir, f"triplane_offset_{offset}.ckpt"))

            # after training the first subset, save pretrained MLP state dict
            # also save all lighting dirs and permuted indices
            if offset_idx == 0:
                for param_group in optimizer.param_groups:
                    if "net" in param_group['name']:
                        final_lr_net = param_group['lr']
                    else:
                        final_lr_net = None
                
                torch.save({
                                'net_state_dict': net.state_dict(),
                                # no longer keep net optimizer state after the final stage of triplane training
                                # TODO: maybe can keep the last net optimizer state somewhere
                                # 'optimizer_state_dict': optimizer.state_dict(),
                                'light_dir_spherical': all_lighting_dirs_spherical,
                                'light_dir_cartesian': all_lighting_dirs_cartesian,
                                'permuted_indices': permuted_indices.tolist(),
                                'lr_net': final_lr_net,
                                'epoch': epoch,
                            }, os.path.join(run_dir, f"pretrained_mlp.ckpt"))
    console_logger.debug(f"Peak memory usage: allocated: {torch.cuda.memory.max_memory_allocated() / 1024**3} reserved: {torch.cuda.memory.max_memory_reserved() / 1024**3}")