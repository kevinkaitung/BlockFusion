from tqdm import tqdm
from easydict import EasyDict as edict
import argparse
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
# import open3d as o3d
# import mcubes, trimesh
import torch
import numpy as np
import os, sys
import json
from visualize_triplane import plot_single_channel

# Add parent directory to sys.path
# TODO: make it more flexible to call timevarying_data_helper anywhere
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from timevarying_data_helper import SampleTimevaryingDataset

def vis_model(net, triplane, n_labels, savedir, oid=0, rank=0):
    os.makedirs(savedir, exist_ok=True)
    for pid in range(n_labels):
        plot_shape(net, triplane, triplane.R * 2, n_labels, 0.0, os.path.join(savedir, f"triplane.ply"),pid, oid)


def save_model(net, triplane, savedir, rank=0):
    os.makedirs(savedir, exist_ok=True)
    torch.save(triplane.state_dict(), os.path.join(savedir, f"tripalne.tar"))


def create_optimizer(net, triplane, config):
    params_to_train = []
    if net is not None:
        params_to_train += [{'name':'net', 'params':net.parameters(), 'lr':config.lr_net}]
    if triplane is not None:
        params_to_train += [{'name':'tri', 'params':triplane.parameters(), 'lr':config.lr_tri}]
    return torch.optim.Adam(params_to_train)

def update_lr(optimizer, epoch, config):
    learning_factor = (np.cos(np.pi * epoch / config.max_iters) + 1.0) * 0.5 * (1 - 0.001) + 0.001
    for param_group in optimizer.param_groups:
        if "net" in param_group['name']:
            param_group['lr'] = config.lr_net * learning_factor
        if "tri" in param_group['name']:
            param_group['lr'] = config.lr_tri * learning_factor

def extract_fields(bound_min, bound_max, resolution, query_func, channel):
    N = 128 # 64. Change it when memory is insufficient!
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution, channel], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs), channel).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def plot_shape(net, triplane, resolution, channel, threshold, savedir, pid, oid):
    u = extract_fields(
        bound_min=[-1.0, -1.0, -1.0],
        bound_max=[ 1.0,  1.0,  1.0],
        resolution=resolution,
        query_func=lambda xyz: -net(triplane(xyz, oid)),
        channel=channel,
    )
    if pid<0:
        u = np.max(u, -1)  # sdf of scene
    else:
        u = u[..., pid]  # sdf of part
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    vertices = vertices / (resolution - 1.0) * 2 - 1
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export(savedir)

def get_triangle_points(obj):
    obj.compute_triangle_normals()
    vertices = np.asarray(obj.vertices)
    triangles = np.asarray(obj.triangles)
    normals = np.asarray(obj.triangle_normals)

    tri_points = torch.from_numpy(vertices[triangles].mean(1)).float()
    tri_normals = torch.from_numpy(normals).float()
    return tri_points, tri_normals

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

            sdf_proxy_2 = nn.Sequential(
                nn.Linear(3, channel * 2), nn.ReLU(),
                nn.Linear(channel * 2, channel * 2),
            )
            torch.nn.init.constant_(sdf_proxy_2[0].bias, 0.0)
            # torch.nn.init.normal_(sdf_proxy[0].weight, 0.0, np.sqrt(2) / np.sqrt(channel))
            torch.nn.init.kaiming_normal_(sdf_proxy_2[0].weight, a=0, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(sdf_proxy_2[2].bias, 0.0)
            # torch.nn.init.normal_(sdf_proxy[2].weight, 0.0, np.sqrt(2) / np.sqrt(channel))
            torch.nn.init.kaiming_normal_(sdf_proxy_2[2].weight, a=0, mode='fan_out', nonlinearity='relu')

            ini_sdf_1 = torch.zeros([1, channel * 2, reso, reso])
            ini_sdf_23 = torch.zeros([2, channel, reso, reso])
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
            # ini_sdf[0] = sdf_proxy(inputx).permute(1, 0).reshape(channel, reso, reso)
            ini_sdf_1[0] = sdf_proxy_2(inputx).permute(1, 0).reshape(channel * 2, reso, reso)
            ini_sdf_23[0] = sdf_proxy(inputy).permute(1, 0).reshape(channel, reso, reso)
            ini_sdf_23[1] = sdf_proxy(inputz).permute(1, 0).reshape(channel, reso, reso)
            # unsqueeze and repeat command just add a new dim and repeat n times at the 1st dim
            self.triplane_1 = torch.nn.Parameter(ini_sdf_1.unsqueeze(0).repeat(self.n, 1, 1, 1, 1) / 4 * 2, requires_grad=True)
            self.triplane_23 = torch.nn.Parameter(ini_sdf_23.unsqueeze(0).repeat(self.n, 1, 1, 1, 1) / 4, requires_grad=True)
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
        # plane_features = self.triplane[oid:oid + 1].view(3, self.C, self.R, self.R)
        plane_features = [self.triplane_1[oid][0:], self.triplane_23[oid][0:1], self.triplane_23[oid][1:2]]
        projected_coordinates = self.project_onto_planes(xyz).unsqueeze(1)
        # feats = F.grid_sample(
        #     plane_features,  # [3,C,R,R]
        #     projected_coordinates.float(),  # [3,1,M,2]
        #     mode="bilinear",
        #     padding_mode="zeros",
        #     align_corners=True
        # )  # [3,C,1,M]
        # feats = feats.permute(0, 3, 2, 1).reshape(3, M, self.C).sum(0)
        
        feats = []
        for i in range(3):
            feats.append(F.grid_sample(
                plane_features[i],  # [1 triplane,C,R,R]
                projected_coordinates[i:i+1].float(),  # [1 triplane,1,M,2]
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True
            ))
        feats = torch.cat([feats[0][0:,:self.C], feats[0][0:,self.C:], feats[1][0:], feats[2][0:]], dim=0).permute(0, 3, 2, 1).reshape(4, M, self.C).sum(0)
        return feats  # [M,C]

    def update_resolution(self, new_reso):
        # old_tri = self.triplane.data.view(self.n * 3, self.C, self.R, self.R)
        # new_tri = F.interpolate(old_tri, size=(new_reso, new_reso), mode='bilinear', align_corners=True)
        # self.R = new_reso
        # self.triplane = torch.nn.Parameter(new_tri.view(self.n, 3, self.C, self.R, self.R), requires_grad=True)
        old_tri_1 = self.triplane_1.data.view(self.n * 1, self.C * 2, self.R, self.R)
        old_tri_23 = self.triplane_23.data.view(self.n * 2, self.C, self.R, self.R)
        new_tri_1 = F.interpolate(old_tri_1, size=(new_reso, new_reso), mode='bilinear', align_corners=True)
        new_tri_23 = F.interpolate(old_tri_23, size=(new_reso, new_reso), mode='bilinear', align_corners=True)
        self.R = new_reso
        self.triplane_1 = torch.nn.Parameter(new_tri_1.view(self.n, 1, self.C * 2, self.R, self.R), requires_grad=True)
        self.triplane_23 = torch.nn.Parameter(new_tri_23.view(self.n, 2, self.C, self.R, self.R), requires_grad=True)
        
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
    # net.load_state_dict(torch.load(config.fixmlp, map_location='cuda'))

    n_timesteps = 90
    # instantiate multiple triplanes (each timestep has its own triplane)
    triplane = [Triplane(
        reso=config.resolution // (2 ** len(config.c2f_scale)),
        channel=config.channel,
        init_type="geo_init",
        objname=None,
    ).cuda() for _ in range(n_timesteps)]
    triplane = nn.ModuleList(triplane)

    optimizer = create_optimizer(net, triplane, config)

    # prepare dataset
    # TODO: tweak to let dataloader return both raw data and timesteps
    train_dataloader = torch.utils.data.DataLoader(
        SampleTimevaryingDataset(
            raw_data_prefix="/media/data/qadwu/volume/vortices",
            raw_data_filename_without_timestep="vorts",
            file_ext="data",
            res=[128, 128, 128],
            n_timesteps=n_timesteps,
            n_channels=1,
            sample_batch_size= 2**10
        ),
        batch_size=config.batch_size,
        shuffle=True)
    # TODO: value range should be retrieve from SampleTimevaryingDataset class (which is more reasonable)
    value_range = 1.0

    for epoch in tqdm(range(1, config.max_iters + 1)):
        
        running_loss = torch.tensor(0.0).cuda()
        
        loss_list = []

        # for debugging
        if epoch % 2000 == 0:
            for dim in range(2):
                plot_single_channel(
                    triplane[50].triplane_1[0][0][dim*config.channel + 16].detach(), 
                    title=f"plane_dim_0_{dim}_epoch_{epoch}",
                    save_path=f"plane_dim_0_{dim}_epoch_{epoch}.png"
                )
            for dim in range(2):
                plot_single_channel(
                    triplane[50].triplane_23[0][dim][16].detach(), 
                    title=f"plane_dim_{dim}_epoch_{epoch}",
                    save_path=f"plane_dim_{dim}_epoch_{epoch}.png"
                )

        if epoch in config.c2f_scale:
            # for debugging
            for dim in range(2):
                plot_single_channel(
                    triplane[50].triplane_1[0][0][dim*config.channel + 16].detach(), 
                    title=f"plane_dim_0_{dim}_reso_{triplane[50].R}",
                    save_path=f"plane_dim_0_{dim}_reso_{triplane[50].R}.png"
                )
            for dim in range(2):
                plot_single_channel(
                    triplane[50].triplane_23[0][dim][16].detach(), 
                    title=f"plane_dim_{dim}_reso_{triplane[50].R}",
                    save_path=f"plane_dim_{dim}_reso_{triplane[50].R}.png"
                )
            new_reso = int(config.resolution / (2 ** (len(config.c2f_scale) - config.c2f_scale.index(epoch) - 1)))
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
            outputs = outputs.squeeze(2)
            # outputs.shape = [90, 1024]
            # targets.shape = [90, 1024]        
            loss = F.mse_loss(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            
        avg_loss = running_loss / len(train_dataloader)
        loss_list.append(avg_loss)
        print(f"Epoch {epoch}, Loss: {avg_loss}, , Reconstruction PSNR: {(20 * torch.log10(value_range / torch.sqrt(avg_loss))):0,.4f}")

        update_lr(optimizer, epoch, config)

    torch.save({
                    'net_state_dict': net.state_dict(),
                    'triplane_state_dict': triplane.state_dict(),
                }, "new_saved_model.ckpt")

    # vis_model(net, triplane, config.n_labels, '.')
    # save_model(net, triplane, '.')

