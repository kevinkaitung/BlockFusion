from easydict import EasyDict as edict
import argparse
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import numpy as np
import os, sys
import json

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
    parser.add_argument('--config', type=str, default='triplane_config_example.json')
    parser.add_argument("--triplane_file_path", type=str, default="logs/Diffusion_on_shadows/20250811-035800/Diffusion_VAE_Reconstructed_triplane.pt")
    args = parser.parse_args()
    
    n_instances = 150
    data_res = [256, 256, 256]

    with open(args.config, 'r') as f:
        config = json.load(f)
    config = edict(config)

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
    
    outputs = net(triplane[0](coords, 0))
    outputs = outputs.reshape(data_res[2], data_res[1], data_res[0])
    outputs.detach().cpu().numpy().astype(np.float32).tofile(f"triplane_reconstructed_volumes.bin")
    