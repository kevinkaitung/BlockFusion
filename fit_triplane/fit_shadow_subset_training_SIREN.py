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

# if torch.cuda.is_available():
#     import tinycudann as tcnn

from networks import NeurCompNet

# Add parent directory to sys.path
# TODO: make it more flexible to call timevarying_data_helper anywhere
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from timevarying_data_helper import SampleShadowVolumesDataset, RandomlyGenerateLightDir, fibonacci_sphere, cartesian_to_spherical_coords

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
    parser.add_argument('--optimizer_type', type=str, default="Adam")
    
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
    console_logger.debug("Optimizer Type: " + args.optimizer_type)
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
    
    epoch = 0
    # for grouping triplanes and their corresponding optimizers
    assert args.n_instances % config.batch_size == 0, "Number of instances must be divisible by batch size"
    offsets = []
    
    # generate subset ckpt first
    for idx in range(args.n_instances // config.batch_size):
        start_idx = idx * config.batch_size
        end_idx = min((idx + 1) * config.batch_size, args.n_instances)

        offsets.append(start_idx)

        # instantiate multiple triplanes (each timestep has its own triplane)
        nets = [
            NeurCompNet(n_input_dims=3, n_output_dims=config.n_labels, bias=False, n_hidden_layers=config.n_layers, n_neurons=config.n_hid, is_residual=True).cuda()
            for _ in range(config.batch_size)]
        nets = nn.ModuleList(nets)
        
        optimizer = create_optimizer(nets, None, config, args.optimizer_type)
        
        torch.save({
            'net_state_dict': nets.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'light_dir_spherical': all_lighting_dirs_spherical[start_idx:end_idx],
            'light_dir_cartesian': all_lighting_dirs_cartesian[start_idx:end_idx],
            'offset': start_idx,
            'end_idx': end_idx,
            # 'lr_net': final_lr_net,
            # 'lr_tri': final_lr_tri,
            'epoch': epoch,
        }, os.path.join(run_dir, f"nets_offset_{start_idx}.ckpt"))

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

    start_epoch = 0
    
    permuted = np.random.permutation(offsets).tolist()
            
    for offset_idx, offset in enumerate(permuted):
        
        subset_model = torch.load(os.path.join(run_dir, f"nets_offset_{offset}.ckpt"))
        
        # start_epoch = subset_model['epoch']
        nets = [
            NeurCompNet(n_input_dims=3, n_output_dims=config.n_labels, bias=False, n_hidden_layers=config.n_layers, n_neurons=config.n_hid, is_residual=True).cuda()
            for _ in range(config.batch_size)]
        nets = nn.ModuleList(nets)
        nets.load_state_dict(subset_model['net_state_dict'])
        
        optimizer = create_optimizer(nets, None, config, args.optimizer_type)
        optimizer.load_state_dict(subset_model['optimizer_state_dict'])
        
        # create DataLoader for this subset
        train_dataloader = torch.utils.data.DataLoader(
        RandomlyGenerateLightDir(
            sampler=sampler,
            n_instances=config.batch_size,
            tfn=args.tfn_file_path,
            sample_batch_size=config.sample_batch_size,
            light_dir_cartesian=subset_model['light_dir_cartesian'],
            # only specify when you need importance sampling on larger gradient points
            # resolution=args.dims,
            # if_gradient=True
        ),
        batch_size=config.batch_size,
        shuffle=True)
        value_range = train_dataloader.dataset.value_range
                    
        
        for epoch in tqdm(range(start_epoch + 1, start_epoch + config.max_iters + 1)):
            
            running_loss = 0.0

            for batch_idx, data in enumerate(train_dataloader):
                
                # data format: tuple(timestep_index, sample_coords, target_values)
                outputs = []
                targets = data[2]
                for timestep, sample_coords in zip(data[0], data[1]):
                    # the output of TCNN MLP would be half type
                    # convert to float type
                    outputs.append(nets[timestep](sample_coords).float())
                    
                # outputs[0].shape = [1024, 1]
                outputs = torch.stack(outputs, dim=0)
                # outputs.shape = [90, 1024, 1]
                # targets.shape = [90, 1024, 1]
                loss = F.mse_loss(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            avg_loss = running_loss / len(train_dataloader)
            PSNR_value = (20 * np.log10(value_range / np.sqrt(avg_loss))).item()
            console_logger.debug(f"Subset {offset}, Epoch {epoch}, Loss: {avg_loss}, Reconstruction PSNR: {(PSNR_value):0,.4f}")
            print(f"Subset {offset}, Epoch {epoch}, Loss: {avg_loss}, , Reconstruction PSNR: {(PSNR_value):0,.4f}")
            tensorboard_writer.add_scalar(f"Loss/Subset_{offset}", avg_loss, epoch)
            tensorboard_writer.add_scalar(f"PSNR/Subset_{offset}", PSNR_value, epoch)
            update_lr(optimizer, epoch, config)
            print(f"Mem Alloc: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB / Max Alloc: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB / Max Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")

        # get the final learning rates of MLP
        for param_group in optimizer.param_groups:
            if "net" in param_group['name']:
                final_lr_net = param_group['lr']
            else:
                final_lr_net = None

        # save this subset training results to disk
        torch.save({
            'net_state_dict': nets.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'light_dir_spherical': subset_model['light_dir_spherical'],
            'light_dir_cartesian': subset_model['light_dir_cartesian'],
            'offset': offset,
            'end_idx': subset_model['end_idx'],
            'lr_net': final_lr_net,
            'epoch': epoch,
        }, os.path.join(run_dir, f"nets_offset_{offset}.ckpt"))

        # save all lighting dirs and permuted indices
        if offset_idx == 0:
            torch.save({
                            'light_dir_spherical': all_lighting_dirs_spherical,
                            'light_dir_cartesian': all_lighting_dirs_cartesian,
                            'permuted_indices': permuted_indices.tolist(),
                        }, os.path.join(run_dir, f"all_lights_info.ckpt"))
    console_logger.debug(f"Peak memory usage: allocated: {torch.cuda.memory.max_memory_allocated() / 1024**3} reserved: {torch.cuda.memory.max_memory_reserved() / 1024**3}")