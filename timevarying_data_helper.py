import os
import numpy as np
import torch

class TimevaryingDataset(torch.utils.data.Dataset):
    def __init__(
        self, raw_data_prefix, raw_data_filename_without_timestep, file_ext, res, n_timesteps, n_channels
    ):
        self.volumes = []
        self.n_timesteps = n_timesteps
        self.n_channels = n_channels
        self.res = res
        for i in range(n_timesteps):
            with open(os.path.join(raw_data_prefix, raw_data_filename_without_timestep+str(i+1)+'.'+file_ext), "rb") as f:
                # f.seek(offset * np.dtype(dtype).itemsize)
                # only read the chunk of the data assigned by the shape
                volume = np.frombuffer(f.read(res[0] * res[1] * res[2] * n_channels * np.dtype(np.float32).itemsize), dtype=np.float32)
                # cast volume data into float32
                # TODO: (the order of res to put into reshape should double check)
                # temporarily ignore since its the cube volume
                volume = volume.astype(np.float32).reshape([res[2], res[1], res[0], n_channels])
                # convert to torch tensor
                volume = torch.from_numpy(volume).cuda()
                # normalize the volume data
                volume = (volume - volume.min()) / (volume.max() - volume.min())
                self.volumes.append(volume)
        self.volumes = torch.stack(self.volumes, dim=0)
        # permute the dimensions to match the expected input shape
        self.volumes = self.volumes.permute(0, 4, 1, 2, 3)  # (n_timesteps, n_channels, res[2], res[1], res[0])
    
    def __getitem__(self, index):
        if index >= self.n_timesteps:
            # need to raise IndexError to avoid infinite loop
            # when directly enumerating the dataset instead of using DataLoader
            # still need this if we use tensor to store our dataset?
            raise IndexError(f"Index {index} out of bounds (n_params={self.n_timesteps})")
        return self.volumes[index]

    def __len__(self):
        return self.n_timesteps


class SampleTimevaryingDataset(torch.utils.data.Dataset):
    def __init__(
        self, raw_data_prefix, raw_data_filename_without_timestep, file_ext, res, n_timesteps, n_channels,
        sample_batch_size=2**10
    ):
        self.volumes = []
        self.n_timesteps = n_timesteps
        self.n_channels = n_channels
        self.res = res
        self.sample_batch_size = sample_batch_size
        for i in range(n_timesteps):
            with open(os.path.join(raw_data_prefix, raw_data_filename_without_timestep+str(i+1)+'.'+file_ext), "rb") as f:
                # f.seek(offset * np.dtype(dtype).itemsize)
                # only read the chunk of the data assigned by the shape
                volume = np.frombuffer(f.read(res[0] * res[1] * res[2] * n_channels * np.dtype(np.float32).itemsize), dtype=np.float32)
                # cast volume data into float32
                # TODO: (the order of res to put into reshape should double check)
                # temporarily ignore since its the cube volume
                volume = volume.astype(np.float32).reshape([res[2], res[1], res[0], n_channels])
                # convert to torch tensor
                volume = torch.from_numpy(volume).cuda()
                # normalize the volume data
                volume = (volume - volume.min()) / (volume.max() - volume.min())
                self.volumes.append(volume)
        self.volumes = torch.stack(self.volumes, dim=0)
        # permute the dimensions to match the expected input shape
        self.volumes = self.volumes.permute(0, 4, 1, 2, 3)  # (n_timesteps, n_channels, res[2], res[1], res[0])
    
    def __getitem__(self, index):
        # generate random coordinates
        # sample_coords: [x_coords, y_coords, z_coords]
        sample_coords = torch.rand([self.sample_batch_size, 3], dtype=torch.float32).cuda()
        # get targets value from the volume at specified timestep
        targets = self.sample(index, sample_coords)
        return index, sample_coords, targets

    def __len__(self):
        return self.n_timesteps

    # sample a batch of data from the volume at index timestep
    def sample(self, index, input):
        with torch.no_grad():
            # Bilinearly filtered lookup from the image. Not super fast,
            # but less than ~20% of the overall runtime of this example.
            shape = self.res

            input = input * torch.tensor([shape[0] - 1, shape[1] - 1, shape[2] - 1], device=input.device).float()
            indices = input.long()
            lerp_weights = input - indices.float()

            x0 = indices[:, 0].clamp(min=0, max=shape[0] - 1)
            y0 = indices[:, 1].clamp(min=0, max=shape[1] - 1)
            z0 = indices[:, 2].clamp(min=0, max=shape[2] - 1)
            x1 = (x0 + 1).clamp(max=shape[0] - 1)
            y1 = (y0 + 1).clamp(max=shape[1] - 1)
            z1 = (z0 + 1).clamp(max=shape[2] - 1)

            # get the volumes at the specified timestep
            # since we only have one channel (scalar field)
            # just use the first channel (index = 0 at second dimension)
            # self.volumes require the access pattern to be [z_coord, y_coord, x_coord]
            c000 = self.volumes[index, 0][z0, y0, x0]
            c010 = self.volumes[index, 0][z0, y1, x0]
            c100 = self.volumes[index, 0][z0, y0, x1]
            c110 = self.volumes[index, 0][z0, y1, x1]
            c001 = self.volumes[index, 0][z1, y0, x0]
            c011 = self.volumes[index, 0][z1, y1, x0]
            c101 = self.volumes[index, 0][z1, y0, x1]
            c111 = self.volumes[index, 0][z1, y1, x1]

            # Trilinear interpolation
            return ((1 - lerp_weights[:,0]) * (1 - lerp_weights[:,1]) * (1 - lerp_weights[:,2]) * c000
                +   (1 - lerp_weights[:,0]) *      lerp_weights[:,1] *  (1 - lerp_weights[:,2]) * c010
                +        lerp_weights[:,0] *  (1 - lerp_weights[:,1]) * (1 - lerp_weights[:,2]) * c100
                +        lerp_weights[:,0] *       lerp_weights[:,1] *  (1 - lerp_weights[:,2]) * c110
                +   (1 - lerp_weights[:,0]) * (1 - lerp_weights[:,1]) *      lerp_weights[:,2] * c001
                +   (1 - lerp_weights[:,0]) *      lerp_weights[:,1] *       lerp_weights[:,2] * c011
                +        lerp_weights[:,0] *  (1 - lerp_weights[:,1]) *      lerp_weights[:,2] * c101
                +        lerp_weights[:,0] *       lerp_weights[:,1] *       lerp_weights[:,2] * c111)


class EncodingWeightDataset(torch.utils.data.Dataset):
    def __init__(
        self, pretrained_weights_info, level
    ):
        self.pretrained_weights_info = pretrained_weights_info
        self.n_params = pretrained_weights_info["tree.n_params"].item()
        base_resolution = pretrained_weights_info["configuration.base_resolution"].item()
        n_features_per_level = pretrained_weights_info["configuration.n_features_per_level"].item()
        per_level_scale = pretrained_weights_info["configuration.per_level_scale"].item()

        # quick and dirty way to slice the encoding weights into two levels
        # TODO: refactor to support all levels
        if level == 0:
            self.offset = 0
            self.length = (base_resolution ** 3) * n_features_per_level
            self.current_resolution = base_resolution
        elif level == 1:
            self.offset = (base_resolution ** 3) * n_features_per_level
            #TODO: check how original code calculate the length if it's not integer
            self.length = int((base_resolution * per_level_scale) ** 3) * n_features_per_level
            self.current_resolution = int(base_resolution * per_level_scale)
        else:
            raise ValueError(f"Invalid level {level}. Must be 0 or 1.")

        self.encoding_weights = []
        for i in range(self.n_params):
            self.encoding_weights.append(pretrained_weights_info[f"weights{i}"][self.offset:self.offset+self.length].reshape(
                self.current_resolution, self.current_resolution, self.current_resolution, n_features_per_level))

        self.encoding_weights = torch.stack(self.encoding_weights, dim=0).cuda()
        # permute the dimensions to match the expected input shape
        self.encoding_weights = self.encoding_weights.permute(0, 4, 1, 2, 3)
        
    def __getitem__(self, index):
        if index >= self.n_params:
            # need to raise IndexError to avoid infinite loop
            # when directly enumerating the dataset instead of using DataLoader
            raise IndexError(f"Index {index} out of bounds (n_params={self.n_params})")
        return self.encoding_weights[index]

    def __len__(self):
        return self.n_params
    
    # since we already specify level in the constructor
    # we don't need to specify level here
    def replace_level_n_weights(self, new_weights):
        results = []
        total_loss = 0.0
        for i in range(self.n_params):
            # replace the encoding weights with the new weights
            running_loss = torch.nn.functional.mse_loss(new_weights[i].permute(1, 2, 3, 0).flatten(), self.pretrained_weights_info[f'weights{i}'][self.offset:self.offset+self.length])
            print(f"{i}: loss {running_loss}")
            total_loss += running_loss
            self.pretrained_weights_info[f"weights{i}"][self.offset:self.offset+self.length] = new_weights[i].permute(1, 2, 3, 0).flatten()
            results.append(self.pretrained_weights_info[f"weights{i}"])
        results = torch.stack(results, dim=0)
        print("total loss", total_loss)
        return results

class LatentWeightDataset(torch.utils.data.Dataset):
    def __init__(
        self, latent_weights, z_shape
    ):
        self.latent_weights = latent_weights
        self.n_params = latent_weights.shape[0]
        
        # need to reshape from flatten input back to VAE recieve
        self.latent_weights = self.latent_weights.reshape([self.n_params, z_shape[0], z_shape[1], z_shape[2], z_shape[3]])
        
    def __getitem__(self, index):
        return self.latent_weights[index], index

    def __len__(self):
        return self.n_params
    
    def get_value_range(self):
        return self.latent_weights.min().item(), self.latent_weights.max().item()

if __name__ == "__main__":
    dataset = TimevaryingDataset(
        raw_data_prefix="/media/data/qadwu/volume/vortices",
        raw_data_filename_without_timestep="vorts",
        file_ext="data",
        res=[128, 128, 128],
        n_timesteps=90,
        n_channels=1
    )
    print("dataset length: ", len(dataset))
    print("dataset shape: ", dataset[0].shape)
    
    pretrained_weights = torch.load("/home/kctung/Projects/instant-vnr-pytorch/logs/hyperinr/debug/run00028/checkpoint-last.ckpt")
    dataset = EncodingWeightDataset(
        pretrained_weights_info=pretrained_weights["model_state_dict"], level=1
    )
    print("dataset length: ", len(dataset))
    print("dataset shape: ", dataset[0].shape)