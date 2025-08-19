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
from pysampler import create_sampler

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


def create_optimizer(net, triplane, config):
    params_to_train = []
    if net is not None:
        params_to_train += [{'name':'net', 'params':net.parameters(), 'lr':config.lr_net}]
    if triplane is not None:
        params_to_train += [{'name':'tri', 'params':triplane.parameters(), 'lr':config.lr_tri}]
    return torch.optim.Adam(params_to_train)

def update_lr(optimizer, epoch, config):
    # TODO: make sure the lr for finetuning is reasonable
    learning_factor = (np.cos(np.pi * epoch / config.max_iters) + 1.0) * 0.5 * (1 - 0.001) + 0.001
    for param_group in optimizer.param_groups:
        if "net" in param_group['name']:
            param_group['lr'] = config.lr_net * learning_factor
        if "tri" in param_group['name']:
            param_group['lr'] = config.lr_tri * learning_factor

class Triplane(nn.Module):
    def __init__(self,
                 n=1,
                 reso=256,
                 channel=32,
                 init_type="geo_init",
                 objname=None,
                 ):
        super().__init__()
        self.n = n
        self.objname = objname
        # assert len(self.objname) == n
        if init_type == "geo_init":
            sdf_proxy = nn.Sequential(
                nn.Linear(3, channel), nn.ReLU(),
                nn.Linear(channel, channel),
            )
            torch.nn.init.constant_(sdf_proxy[0].bias, 0.0)
            # torch.nn.init.normal_(sdf_proxy[0].weight, 0.0, np.sqrt(2) / np.sqrt(channel))
            torch.nn.init.kaiming_normal_(sdf_proxy[0].weight, a=0, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(sdf_proxy[2].bias, 0.0)
            # torch.nn.init.normal_(sdf_proxy[2].weight, 0.0, np.sqrt(2) / np.sqrt(channel))
            torch.nn.init.kaiming_normal_(sdf_proxy[2].weight, a=0, mode='fan_out', nonlinearity='relu')

            ini_sdf = torch.zeros([3, channel, reso, reso])
            # create XY, XZ, YZ planes grid coordinates to initialize the triplane's weights
            # since grid_sample expects the input to be in the range [-1, 1]
            # we create a grid of points in the range [-1, 1] for each plane
            X = torch.linspace(-1.0, 1.0, reso)
            (U, V) = torch.meshgrid(X, X, indexing="ij")
            Z = torch.zeros(reso, reso)
            inputx = torch.stack([Z, U, V], -1).reshape(-1, 3)
            inputy = torch.stack([U, Z, V], -1).reshape(-1, 3)
            inputz = torch.stack([U, V, Z], -1).reshape(-1, 3)
            # use permute to make the channel dimension as first dimension, batch size as second dimension
            # and reshape back to grid format
            ini_sdf[0] = sdf_proxy(inputx).permute(1, 0).reshape(channel, reso, reso)
            ini_sdf[1] = sdf_proxy(inputy).permute(1, 0).reshape(channel, reso, reso)
            ini_sdf[2] = sdf_proxy(inputz).permute(1, 0).reshape(channel, reso, reso)
            # unsqueeze and repeat command just add a new dim and repeat n times at the 1st dim
            self.triplane = torch.nn.Parameter(ini_sdf.unsqueeze(0).repeat(self.n, 1, 1, 1, 1) / 3, requires_grad=True)
        elif init_type == "zero_init":
            self.triplane = torch.nn.Parameter(torch.zeros([self.n, 3, channel, reso, reso]), requires_grad=True)

        self.R = reso
        self.C = channel
        # construct the matrix to project points onto the three planes
        # but the basis vectors are actually used to project from 2D space to 3D space
        # so, we will get the inverse of the matrix later to project from 3D space to 2D space
        # three matrices are used to project points onto the XY, XZ, and YZ planes respectively
        self.register_buffer("plane_axes", torch.tensor(
            [[[0, 1, 0],
              [1, 0, 0],
              [0, 0, 1]],
             [[0, 0, 1],
              [1, 0, 0],
              [0, 1, 0]],
             [[0, 1, 0],
              [0, 0, 1],
              [1, 0, 0]]], dtype=torch.float32)
                             )

        # xy xz yz

    def project_onto_planes(self, xyz):
        # xyz shape: [sample_batch_size, 3 (xyz coords)]
        M, _ = xyz.shape
        # expand xyz 3 times at the first dimension, possibly for all 3 triplanes
        xyz = xyz.unsqueeze(0).expand(3, -1, -1).reshape(3, M, 3)
        # since we are projecting from 3D space to 2D space,
        # get the inverse of the projection matrix (plane_axes)
        inv_planes = torch.linalg.inv(self.plane_axes).reshape(3, 3, 3)
        # inv_planes:
        # [[[0., 1., 0.],
        #  [1., 0., 0.],
        #  [0., 0., 1.]],
        # [[0., 1., 0.],
        #  [0., 0., 1.],
        #  [1., 0., 0.]],
        # [[0., 0., 1.],
        #  [1., 0., 0.],
        #  [0., 1., 0.]]]
        # then get the product of the input points and the inverse projection matrix
        projections = torch.bmm(xyz, inv_planes)
        # projections[0,:,2] stores [y_in_org_3D_space, x_in_org_3D_space]
        # projections[1,:,2] stores [z_in_org_3D_space, x_in_org_3D_space]
        # projections[2,:,2] stores [y_in_org_3D_space, z_in_org_3D_space]
        # since grid_sample expects input grid values in the range [-1, 1]
        # normalize from [0, 1] to [-1, 1]
        return 2.0 * projections[..., :2] - 1.0  # [3, M, 2]

    def forward(self, xyz, oid):
        # xyz shape: [sample_batch_size, 3 (xyz coords)]
        # pts: [M,3]
        M, _ = xyz.shape
        plane_features = self.triplane[oid:oid + 1].view(3, self.C, self.R, self.R)
        projected_coordinates = self.project_onto_planes(xyz).unsqueeze(1)
        feats = F.grid_sample(
            plane_features,  # [3,C,R,R]
            projected_coordinates.float(),  # [3,1,M,2]
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )  # [3,C,1,M]
        feats = feats.permute(0, 3, 2, 1).reshape(3, M, self.C).sum(0)
        return feats  # [M,C]

    def update_resolution(self, new_reso):
        old_tri = self.triplane.data.view(self.n * 3, self.C, self.R, self.R)
        new_tri = F.interpolate(old_tri, size=(new_reso, new_reso), mode='bilinear', align_corners=True)
        self.R = new_reso
        self.triplane = torch.nn.Parameter(new_tri.view(self.n, 3, self.C, self.R, self.R), requires_grad=True)


class Network(nn.Module):
    def __init__(self,
                 d_in=32,
                 d_hid=128,
                 n_layers=3,
                 d_out=6,
                 init_type="geo_init",
                 weight_norm=True,
                 bias=0.5
                 ):
        super().__init__()
        dims = [d_in] + [d_hid for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            in_dim = dims[l]
            out_dim = dims[l + 1]
            lin = nn.Linear(in_dim, out_dim)

            if init_type == "geo_init":
                # last layer use different init strategy (xavier)
                if l == self.num_layers - 2:
                    torch.nn.init.xavier_normal_(lin.weight)
                    # TODO: why original implementation only initialize last layer's bias as bias?
                    # (because hidden layer initialize as 0.0)
                    torch.nn.init.constant_(lin.bias, bias)
                # use kaiming init for hidden layers (which is suitable for ReLU)
                else:
                    torch.nn.init.kaiming_normal_(lin.weight, a=0, mode='fan_out', nonlinearity='relu')
                    torch.nn.init.constant_(lin.bias, 0.0)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            # use setattr to dynamically create member variables of the class
            # e.g. setattr(self, "lin0", lin) for the first layer
            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()

    def forward(self, feats):
        x = feats
        for l in range(0, self.num_layers - 1):
            # get corresponding linear layer and apply it
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            # apply activation function for hidden layers
            if l < self.num_layers - 2:
                x = self.activation(x)
        return x

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_timevarying.json')
    parser.add_argument("--expname", type=str, default="finetune_VAE_recon_triplanes", help="Experiment name")
    parser.add_argument("--description", type=str, default="", help="Description to experiment")
    parser.add_argument('--resume_training_model', type=str, default=None)
    parser.add_argument('--only_finetune_mlp', action='store_true')
    
    parser.add_argument('--dims', type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--raw_data_file_path', type=str, default="/media/data/qadwu/volume/vortices/vorts1.data")
    parser.add_argument('--tfn_file_path', type=str, default="/home/kctung/Projects/instant-vnr-pytorch/bindings/ovr/data/configs/vorts_shadow.json")
    parser.add_argument('--n_instances', type=int, default=150, help="Number of shadow volumes to generate")
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
    if args.resume_training_model:
        console_logger.debug("Resume Training Model Path: " + args.resume_training_model)
    console_logger.debug("Config File Name: " + args.config)
    console_logger.debug("Only finetune mlp: " + str(args.only_finetune_mlp))

    # if args.only_finetune_mlp:
    #     console_logger.debug("Only finetune mlp: True")
    # else:
        # console_logger.debug("Only finetune mlp: False")
    
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
    # net.load_state_dict(torch.load(config.fixmlp, map_location='cuda'))

    # instantiate multiple triplanes (each timestep has its own triplane)
    triplane = [Triplane(
        reso=config.resolution // (2 ** len(config.c2f_scale)),
        channel=config.channel,
        init_type="geo_init",
        objname=None,
    ).cuda() for _ in range(args.n_instances)]
    triplane = nn.ModuleList(triplane)

    # for fine tune model (both for only finetune MLP, finetune triplanes+MLP, and resume training from original triplane overfitting training)
    if args.resume_training_model:
        loaded_model = torch.load(args.resume_training_model)
        net.load_state_dict(loaded_model['net_state_dict'])
        triplane.load_state_dict(loaded_model['triplane_state_dict'])
        # TODO: make sure whether I should use optimizer from pretrained triplane?
        if args.only_finetune_mlp:
            optimizer = create_optimizer(net, None, config)
        else:
            optimizer = create_optimizer(net, triplane, config)
    else:
        optimizer = create_optimizer(net, triplane, config)

    sampler = create_sampler("structuredRegular", "cuda", dims=args.dims, dtype=args.dtype, n_channels=1, filename=args.raw_data_file_path)

    # prepare dataset
    train_dataloader = torch.utils.data.DataLoader(
        RandomlyGenerateLightDir(
            sampler=sampler,
            n_instances=args.n_instances,
            tfn=args.tfn_file_path,
            sample_batch_size=config.sample_batch_size
        ),
        batch_size=config.batch_size,
        shuffle=True)
    
    value_range = train_dataloader.dataset.value_range
    
    if "resume_iter" in config:
        start_iter = config.resume_iter
    else:
        start_iter = 0

    for epoch in tqdm(range(start_iter+1, config.max_iters + 1)):
        
        running_loss = torch.tensor(0.0).cuda()
        
        loss_list = []

        # for debugging
        if epoch % 2000 == 0:
            for dim in range(3):
                plot_single_channel(
                    triplane[50].triplane[0][dim][16].detach(), 
                    title=f"plane_dim_{dim}_epoch_{epoch}",
                    save_path=os.path.join(run_dir, f"plane_dim_{dim}_epoch_{epoch}.png")
                )

        if epoch in config.c2f_scale:
            new_reso = int(config.resolution / (2 ** (len(config.c2f_scale) - config.c2f_scale.index(epoch) - 1)))
            # for debugging
            for dim in range(3):
                plot_single_channel(
                    triplane[50].triplane[0][dim][16].detach(), 
                    title=f"plane_dim_{dim}_reso_{new_reso}",
                    save_path=os.path.join(run_dir, f"plane_dim_{dim}_reso_{new_reso}.png")
                )
            for tri in triplane:
                tri.update_resolution(new_reso)
            optimizer = create_optimizer(net, triplane, config)
            update_lr(optimizer, epoch - 1, config)
            torch.cuda.empty_cache()

        for batch_idx, data in enumerate(train_dataloader):
            # data format: tuple(timestep_index, sample_coords, target_values)
            outputs = []
            targets = data[2]
            for timestep, sample_coords in zip(data[0], data[1]):
                outputs.append(net(triplane[timestep](sample_coords, 0)))
            # outputs[0].shape = [1024, 1]
            outputs = torch.stack(outputs, dim=0)
            # outputs.shape = [90, 1024, 1]
            # targets.shape = [90, 1024, 1]
            loss = F.mse_loss(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            
        avg_loss = running_loss / len(train_dataloader)
        loss_list.append(avg_loss)
        PSNR_value = (20 * torch.log10(value_range / torch.sqrt(avg_loss))).item()
        console_logger.debug(f"Epoch {epoch}, Loss: {avg_loss.item()}, , Reconstruction PSNR: {(PSNR_value):0,.4f}")
        print(f"Epoch {epoch}, Loss: {avg_loss.item()}, , Reconstruction PSNR: {(PSNR_value):0,.4f}")
        tensorboard_writer.add_scalar("Loss/Train", avg_loss.item(), epoch)
        tensorboard_writer.add_scalar("Loss/Train_PSNR", PSNR_value, epoch)
        update_lr(optimizer, epoch, config)

    for param_group in optimizer.param_groups:
        if "net" in param_group['name']:
            final_lr_net = param_group['lr']
        if "tri" in param_group['name']:
            final_lr_tri = param_group['lr']
        else:
            final_lr_tri = None

    torch.save({
                    'net_state_dict': net.state_dict(),
                    'triplane_state_dict': triplane.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'light_dir_spherical': train_dataloader.dataset.light_dir_spherical.tolist(),
                    'light_dir_cartesian': train_dataloader.dataset.light_dir_cartesian.tolist(),
                    'lr_net': final_lr_net,
                    'lr_tri': final_lr_tri,
                    'epoch': epoch,
                }, os.path.join(run_dir, f"triplane_model_{epoch}.ckpt"))