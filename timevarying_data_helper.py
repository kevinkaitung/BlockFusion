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