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

# Add parent directory to sys.path
# TODO: make it more flexible to call timevarying_data_helper anywhere
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from visualize_triplane import plot_single_channel
from timevarying_data_helper import SampleShadowVolumesDataset, RandomlyGenerateLightDir
from fit_shadow_randomly_generate import Triplane, MLP_TCNN, Network

check_plane_idx = 40
vis_triplane_freq = 2000

def create_optimizer(net, triplane, config, optimizer_type):
    params_to_train = []
    if net is not None:
        params_to_train += [{'name':'net', 'params':net.parameters(), 'lr':config.lr_net}]
    if triplane is not None:
        params_to_train += [{'name':'tri', 'params':triplane.parameters(), 'lr':config.lr_tri}]
    if optimizer_type == "Adam":
        return torch.optim.Adam(params_to_train)
    elif optimizer_type == "SGD":
        return torch.optim.SGD(params_to_train)
    else:
        raise RuntimeError(f"{optimizer_type} optimizer not supported!")

def update_lr(optimizer, epoch, config):
    # here assume we train from epoch 0, so simply take fraction current epoch over total epoch
    # since we use pretrained MLP to generate more triplane
    learning_factor = (np.cos(np.pi * epoch / config.max_iters) + 1.0) * 0.5 * (1 - 0.001) + 0.001
    for param_group in optimizer.param_groups:
        # might not need this anymore because pretrained MLP only used to optimize triplanes
        # if "net" in param_group['name']:
        #     param_group['lr'] = config.lr_net * learning_factor
        if "tri" in param_group['name']:
            param_group['lr'] = config.lr_tri * learning_factor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_timevarying.json')
    parser.add_argument("--expdir", type=str, default="logs/shadows_subset_training_chameleon/20251003-002117")
    parser.add_argument("--description", type=str, default="", help="Description to experiment")
    
    # parser.add_argument('--only_finetune_mlp', action='store_true')
    # parser.add_argument('--freeze_mlp', action='store_true')
    parser.add_argument('--optimizer_type', type=str, default="Adam")
    parser.add_argument('--use_native_mlp', action='store_true')
    
    parser.add_argument('--dims', type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--raw_data_file_path', type=str, default="/media/data/qadwu/volume/vortices/vorts1.data")
    parser.add_argument('--tfn_file_path', type=str, default="/home/kctung/Projects/instant-vnr-pytorch/bindings/ovr/data/configs/vorts_shadow.json")
    # parser.add_argument('--n_instances', type=int, default=150, help="Number of shadow volumes to generate")
    parser.add_argument("--sampled_lighting_dirs_path", type=str, default=None, help="File Path to lighting dirs list")
    args = parser.parse_args()

    # create directory for saving logs
    run_dir = args.expdir
    logging_file_md = 'a'
    
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
    console_logger.debug("Config File Name: " + args.config)
    console_logger.debug("Optimizer Type: " + args.optimizer_type)
    console_logger.debug("Use Native MLP: " + str(args.use_native_mlp))
    console_logger.debug("Path to Raw Data File: " + args.raw_data_file_path)
    console_logger.debug("Path to TFN Data File: " + args.tfn_file_path)
    console_logger.debug("Sampled Lighting Directions Json File Path: " + str(args.sampled_lighting_dirs_path))

    with open(args.config, 'r') as f:
        config = json.load(f)
    config = edict(config)
    # assert len(config.fixmlp) > 0
    
    sampler = create_sampler("structuredRegular", "cuda", dims=args.dims, dtype=args.dtype, n_channels=1, filename=args.raw_data_file_path)

    with open(args.sampled_lighting_dirs_path, 'r') as f:
        sampled_lighting_dirs = json.load(f)
    # in cartesian
    sampled_lighting_dirs = np.array(sampled_lighting_dirs).reshape(-1, 3)
    
    n_instances = sampled_lighting_dirs.shape[0]
    console_logger.debug("Number of instances: " + str(n_instances))

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
                    activation="ReLU", output_activation="None")
    
    loaded_pretrained_mlp = torch.load(os.path.join(run_dir, "pretrained_mlp.ckpt"))
    net.load_state_dict(loaded_pretrained_mlp['net_state_dict'])
    
    # TODO: might need to support create triplanes by batch later
    # like in the subset training script
    # instantiate multiple triplanes (each timestep has its own triplane)
    triplane = [Triplane(
        reso=config.resolution // (2 ** len(config.c2f_scale)),
        channel=config.channel,
        init_type="geo_init",
        objname=None,
    ).cuda() for _ in range(n_instances)]
    print(f"Number of instances: {n_instances}")
    triplane = nn.ModuleList(triplane)

    optimizer = create_optimizer(None, triplane, config, args.optimizer_type)
    # freeze MLP
    for p in net.parameters():
        p.requires_grad = False
    
    # prepare dataset
    train_dataloader = torch.utils.data.DataLoader(
        RandomlyGenerateLightDir(
            sampler=sampler,
            n_instances=n_instances,
            tfn=args.tfn_file_path,
            sample_batch_size=config.sample_batch_size,
            light_dir_cartesian=sampled_lighting_dirs.tolist()
        ),
        batch_size=config.batch_size,
        shuffle=True)
    
    value_range = train_dataloader.dataset.value_range
    
    start_iter = 0

    for epoch in tqdm(range(start_iter+1, config.max_iters + 1)):
        
        running_loss = 0.0
        
        # loss_list = []

        # for debugging
        if epoch % vis_triplane_freq == 0:
            for dim in range(3):
                plot_single_channel(
                    triplane[check_plane_idx].triplane[0][dim][16].detach(), 
                    title=f"plane_additional_dim_{dim}_epoch_{epoch}",
                    save_path=os.path.join(run_dir, f"plane_additional_dim_{dim}_epoch_{epoch}.png")
                )

        if epoch in config.c2f_scale:
            new_reso = int(config.resolution / (2 ** (len(config.c2f_scale) - config.c2f_scale.index(epoch) - 1)))
            # for debugging
            for dim in range(3):
                plot_single_channel(
                    triplane[check_plane_idx].triplane[0][dim][16].detach(), 
                    title=f"plane_additional_dim_{dim}_reso_{new_reso}",
                    save_path=os.path.join(run_dir, f"plane_additional_dim_{dim}_reso_{new_reso}.png")
                )
            for tri in triplane:
                tri.update_resolution(new_reso)
            optimizer = create_optimizer(None, triplane, config, args.optimizer_type)
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
                # # Use checkpoint to trade compute for memory
                # def checkpoint_fn(coords):
                #     encoded = triplane[timestep](coords, 0)
                #     return net(encoded)
                # # Checkpoint saves memory by not storing intermediate activations
                # output = checkpoint(checkpoint_fn, sample_coords, use_reentrant=False)
                # outputs.append(output)
            
            # outputs[0].shape = [1024, 1]
            outputs = torch.stack(outputs, dim=0)
            # outputs.shape = [90, 1024, 1]
            # targets.shape = [90, 1024, 1]
            loss = F.mse_loss(outputs, targets)

            # deleting targets and outputs might not clean up much memory
            # most of the memory might be consumed by the intermediate results of forward pass
            del targets, outputs
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            # print(f"Before backward Mem Alloc: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB / Max Alloc: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB / Max Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
            loss.backward()
            # print(f"Before step Mem Alloc: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB / Max Alloc: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB / Max Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
            optimizer.step()
            # print(f"After step Mem Alloc: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB / Max Alloc: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB / Max Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
            
            running_loss += loss.item()
            del loss
            torch.cuda.empty_cache()
            
        avg_loss = running_loss / len(train_dataloader)
        # loss_list.append(avg_loss)
        PSNR_value = (20 * np.log10(value_range / np.sqrt(avg_loss))).item()
        console_logger.debug(f"Epoch {epoch}, Loss: {avg_loss}, , Reconstruction PSNR: {(PSNR_value):0,.4f}")
        print(f"Epoch {epoch}, Loss: {avg_loss}, , Reconstruction PSNR: {(PSNR_value):0,.4f}")
        tensorboard_writer.add_scalar("Loss/Train", avg_loss, epoch)
        tensorboard_writer.add_scalar("Loss/Train_PSNR", PSNR_value, epoch)
        update_lr(optimizer, epoch, config)
        print(f"Mem Alloc: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB / Max Alloc: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB / Max Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")

    for param_group in optimizer.param_groups:
        if "net" in param_group['name']:
            final_lr_net = param_group['lr']
        else:
            final_lr_net = None
        if "tri" in param_group['name']:
            final_lr_tri = param_group['lr']
        else:
            final_lr_tri = None

    torch.save({
                    'net_state_dict': net.state_dict(),
                    'triplane_state_dict': triplane.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    'light_dir_spherical': train_dataloader.dataset.light_dir_spherical.tolist(),
                    'light_dir_cartesian': train_dataloader.dataset.light_dir_cartesian.tolist(),
                    'lr_net': final_lr_net,
                    'lr_tri': final_lr_tri,
                    'epoch': epoch,
                }, os.path.join(run_dir, f"triplane_model_additional.ckpt"))
    console_logger.debug(f"Peak memory usage: allocated: {torch.cuda.memory.max_memory_allocated() / 1024**3} reserved: {torch.cuda.memory.max_memory_reserved() / 1024**3}")