import os
import numpy as np
import torch
import glob
import json
try:
    from pysampler import decode_shadow
except ImportError:
    pass

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
            
class ShadowVolumesMetaDataset(torch.utils.data.Dataset):
    def __init__(
        self, raw_data_dir, raw_data_filename_prefix, file_ext, n_instances
    ):
        self.n_instances = n_instances
        self.file_names = []
        self.lighting_dirs = []
        
        # Find all files starting with "shadow" and ending with your file extension
        pattern = os.path.join(raw_data_dir, f"{raw_data_filename_prefix}*.{file_ext}")
        file_list = sorted(glob.glob(pattern))  # sorted ensures consistent order (would be lexicographic order)
        for filepath in file_list:
            with open(filepath, "r") as f:
                shadow_meta = json.load(f)
                direction=shadow_meta["view"]["lightSource"]["position"]
                dir_tensor = torch.tensor([direction["x"], direction["y"], direction['z']], dtype=torch.float32).cuda()
                self.file_names.append(filepath)
                self.lighting_dirs.append(dir_tensor)
        
        self.lighting_dirs = torch.stack(self.lighting_dirs, dim=0)
        
        
    def __getitem__(self, index):
        # if index >= self.n_instances:
        #     raise IndexError(f"Index {index} out of bounds (n_params={self.n_instances})")
        return self.lighting_dirs[index]

    def __len__(self):
        return self.n_instances

class ShadowLightingDirectionsDataset(torch.utils.data.Dataset):
    def __init__(
        self, lighting_dirs
    ):
        self.n_instances = len(lighting_dirs)
        # lighting_dirs would be 2D python list originally, turn it to torch tensor
        self.lighting_dirs = torch.tensor(lighting_dirs).cuda()
    
    def __getitem__(self, index):
        return self.lighting_dirs[index]

    def __len__(self):
        return self.n_instances
    
    def get_all_light_dirs_list(self):
        return self.lighting_dirs.tolist()

class ShadowVolumesDataset(torch.utils.data.Dataset):
    def __init__(
        self, raw_data_dir, raw_data_filename_prefix, file_ext, res, n_instances, n_channels
    ):
        self.volumes = []
        self.n_instances = n_instances
        self.n_channels = n_channels
        self.res = res
        self.data_min = 0.0
        self.data_max = 1.0
        self.value_range = self.data_max - self.data_min
        
        # Find all files starting with "shadow" and ending with your file extension
        pattern = os.path.join(raw_data_dir, f"{raw_data_filename_prefix}*.{file_ext}")
        file_list = sorted(glob.glob(pattern))  # sorted ensures consistent order (would be lexicographic order)
        for filepath in file_list:
            with open(filepath, "rb") as f:
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
        if index >= self.n_instances:
            raise IndexError(f"Index {index} out of bounds (n_params={self.n_instances})")
        return self.volumes[index]

    def __len__(self):
        return self.n_instances

class SampleShadowVolumesDataset(torch.utils.data.Dataset):
    def __init__(
        self, raw_data_dir, raw_data_filename_prefix, file_ext, res, n_instances, n_channels,
        sample_batch_size=2**10
    ):
        self.volumes = []
        self.n_instances = n_instances
        self.n_channels = n_channels
        self.res = res
        self.sample_batch_size = sample_batch_size
        self.data_min = 0.0
        self.data_max = 1.0
        self.value_range = self.data_max - self.data_min
        
        # Find all files starting with "shadow" and ending with your file extension
        pattern = os.path.join(raw_data_dir, f"{raw_data_filename_prefix}*.{file_ext}")
        file_list = sorted(glob.glob(pattern))  # sorted ensures consistent order (would be lexicographic order)
        for filepath in file_list:
            with open(filepath, "rb") as f:
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
        return self.n_instances

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

# this function probably transforms normalized value range (0~1) into narrower value range (i.e., 0.05~0.45)
# so we would only generate the shadow volumes under some specific light directions (not from all directions)
def param2lightdir(param):
    if not isinstance(param, np.ndarray):
        raise ValueError("param must be a numpy.ndarray")
    theta = (param[..., 0] - 0.5) / 4.0 + 0.50
    phi   = (param[..., 1] - 0.5) / 2.5 + 0.25
    return np.stack([theta, phi], axis=-1)

# double check the correctness
# def spherical_to_cartesian_coords(spherical_coords):
#     theta, phi = spherical_coords[..., 0], spherical_coords[..., 1]
#     theta = 2.0 * np.pi * theta
#     phi = 1.0 * np.pi * phi
    
#     x = 1.0 * np.sin(theta) * np.cos(phi)
#     y = 1.0 * np.sin(theta) * np.sin(phi)
#     z = 1.0 * np.cos(theta)
#     cartesian_coords = np.stack([x, y, z], axis=-1)
#     norms = np.linalg.norm(cartesian_coords, axis=-1, keepdims=True)
#     cartesian_coords = cartesian_coords / norms
#     return cartesian_coords

def spherical_to_cartesian_coords(spherical_coords):
    theta, phi = spherical_coords[..., 0], spherical_coords[..., 1]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=-1)

def cartesian_to_spherical_coords(cartesian_coords):
    x, y, z = cartesian_coords[..., 0], cartesian_coords[..., 1], cartesian_coords[..., 2]
    r = np.linalg.norm(cartesian_coords, axis=-1)
    theta = np.arctan2(y, x)             # azimuth
    phi = np.arccos(z / r)               # polar
    return np.stack([theta, phi], axis=-1)

def spherical_coords_radiance_to_normalized(spherical_coords):
    theta, phi = spherical_coords[..., 0], spherical_coords[..., 1]
    theta = (theta % (2.0 * np.pi)) / (2.0 * np.pi)
    phi = phi / np.pi
    return np.stack([theta, phi], axis=-1)

# generate lighting directions based on fibonacci sphere
def fibonacci_sphere(samples=1000, randomize=True):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * samples

    points = []
    offset = 2.0 / samples
    increment = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y * y)

        phi = ((i + rnd) % samples) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        points.append([x, y, z])

    return np.array(points)

class RandomlyGenerateLightDir(torch.utils.data.Dataset):
    def __init__(
        self, sampler, n_instances, tfn, sample_batch_size=2**10, light_dir_spherical=None, light_dir_cartesian=None,
        resolution=[None, None, None], if_gradient=False
    ):
        self.sampler = sampler
        self.n_instances = n_instances
        self.tfn = tfn
        self.sample_batch_size = sample_batch_size
        if light_dir_spherical is None and light_dir_cartesian is None:
            # randomly generate n_instances light dir
            # self.light_dir_spherical = np.random.rand(n_instances, 2)
            self.light_dir_cartesian = fibonacci_sphere(n_instances)
            self.light_dir_spherical = cartesian_to_spherical_coords(self.light_dir_cartesian)
        else:
            if light_dir_spherical:
                self.light_dir_spherical = np.array(light_dir_spherical)
                self.light_dir_cartesian = spherical_to_cartesian_coords(self.light_dir_spherical)
            elif light_dir_cartesian:
                self.light_dir_cartesian = np.array(light_dir_cartesian)
                self.light_dir_spherical = cartesian_to_spherical_coords(self.light_dir_cartesian)
        # convert spherical coords from radiance unit to normalized value (0~1) for using in shadow sampler
        self.light_dir_spherical_normalized = spherical_coords_radiance_to_normalized(self.light_dir_spherical)

        self.if_gradient = if_gradient
        if if_gradient:
            self.selected_coord_groups, self.selected_value_groups = self.calculate_gradient(resolution)
        
        # for debug:
        # print("original spherical coords in radiance:", self.light_dir_spherical[:50])
        # print("transformed spherical coords in normalized value:", self.light_dir_spherical_normalized[:50])
        self.data_min = 0.0
        self.data_max = 1.0
        self.value_range = self.data_max - self.data_min
        
    def __getitem__(self, index):
        # generate random coordinates
        # sample_coords: [x_coords, y_coords, z_coords]
        
        if self.if_gradient:
            selected_idx = torch.randperm(self.selected_coord_groups[index].shape[0])[:self.sample_batch_size]
            selected_coords = self.selected_coord_groups[index][selected_idx].cuda()
            
            # uniformly sample over the whole volume domain
            sample_coords = torch.rand([1024, 3], dtype=torch.float32, device="cuda")
            sample_coords = torch.cat([sample_coords, selected_coords], dim=0)
            
            targets = torch.empty((sample_coords.shape[0], 1), dtype=torch.float32, device="cuda")
        else:
            # uniformly sample over the whole volume domain
            sample_coords = torch.rand([self.sample_batch_size, 3], dtype=torch.float32, device="cuda")
            targets = torch.empty((self.sample_batch_size, 1), dtype=torch.float32, device="cuda")
        
        decode_shadow(self.sampler, sample_coords, targets, self.light_dir_spherical_normalized[index], self.tfn)
        return index, sample_coords, targets

    def __len__(self):
        return self.n_instances
    
    def decode_ith_shadow_volume(self, index, sample_coords):
        targets = torch.empty((sample_coords.shape[0], 1), dtype=torch.float32, device="cuda")
        decode_shadow(self.sampler, sample_coords, targets, self.light_dir_spherical_normalized[index], self.tfn)
        return index, sample_coords, targets
    
    # extended function for calculating gradients and gradients' norm
    def calculate_gradient(self, resolution):
        
        from fit_triplane.data_distribution_analyze import generate_coords_chunks
        from fit_triplane.calculate_gradient import calculate_gradient
        
        chunk_size = 65536*192
    
        # grad_norms = []
        selected_coord_groups = []
        selected_value_groups = []
        # need to decode the volume first
        for idx in range(self.n_instances):
            targets = []
            for coord_chunk in generate_coords_chunks(resolution, chunk_size):
                target = torch.empty([coord_chunk.shape[0], 1]).float().cuda()
                decode_shadow(self.sampler, coord_chunk, target, self.light_dir_spherical_normalized[idx], self.tfn)
                targets.append(target.cpu())
            targets = torch.cat(targets, dim=0)
            targets = targets.reshape([resolution[2], resolution[1], resolution[0]])
            
            ### section to generate coords based on grad norm
            # currently only use grad norm
            # _, grad_norm = calculate_gradient(targets)
            # # grad_norms.append(grad_norm)
            # grad_norm_thres = 0.5
            # selected_coords = torch.nonzero(grad_norm > grad_norm_thres)
            # print(f"instance {idx}: {selected_coords.shape[0]} points passing the gradient norm threshold")
            # # swap the 1st and 3rd col ((z, y, x) -> (x, y, z))
            # selected_coords[:, [0, 2]] = selected_coords[:, [2, 0]]
            # selected_value_groups.append(targets[selected_coords[:, 2].int(), selected_coords[:, 1].int(), selected_coords[:, 0].int()])
            # # NOTE: should normalize selected coords back to 0 to 1!!!
            # selected_coords = selected_coords / torch.tensor([resolution[0] - 1, resolution[1] - 1, resolution[2] - 1], dtype=torch.float32)
            # selected_coord_groups.append(selected_coords)
            ### section end
            
            ### section to uniformly generate coords
            selected_coords = torch.rand([500000, 3], dtype=torch.float32) * torch.tensor([resolution[0] - 1, resolution[1] - 1, resolution[2] - 1], dtype=torch.float32)
            selected_value_groups.append(targets[selected_coords[:, 2].int(), selected_coords[:, 1].int(), selected_coords[:, 0].int()])
            # NOTE: should normalize selected coords back to 0 to 1!!!
            selected_coords = selected_coords / torch.tensor([resolution[0] - 1, resolution[1] - 1, resolution[2] - 1], dtype=torch.float32)
            selected_coord_groups.append(selected_coords)
            ### section end
            
        return selected_coord_groups, selected_value_groups
        

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
        self, latent_weights, z_shape, min, max
    ):
        self.latent_weights = latent_weights
        # self.n_params = latent_weights.shape[0]
        self.n_params = len(latent_weights)
        
        self.min = min
        self.max = max
        
        # need to reshape from flatten input back to VAE recieve
        # self.latent_weights = self.latent_weights.reshape([self.n_params] + [z_shape[idx] for idx in range(len(z_shape))])
        for idx in range(self.n_params):
            self.latent_weights[idx] = self.latent_weights[idx].reshape([z_shape[idx] for idx in range(len(z_shape))])
        
    def __getitem__(self, index):
        return self.latent_weights[index], index

    def __len__(self):
        return self.n_params
    
    def get_value_range(self):
        # return self.latent_weights.min().item(), self.latent_weights.max().item()
        return self.min, self.max

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