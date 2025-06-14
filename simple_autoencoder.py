import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # (B, 32, 128, 128)
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            # (B, 128, 128, 128)
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            # (B, 512, 64, 64)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            # (B, 512, 32, 32)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            # (B, 1024, 16, 16)
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            # (B, 1024, 8, 8)
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            
            # (B, 1024, 4, 4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            # (B, 1024, 8, 8)
            nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            # (B, 1024, 16, 16)
            nn.ConvTranspose2d(in_channels=1024, out_channels=4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(4),
            nn.SiLU(),
            # (B, 4, 32, 32)
        )

        # Decoder
        self.decoder = nn.Sequential(
            # (B, 4, 32, 32)
            nn.Conv2d(in_channels=4, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            # (B, 512, 32, 32)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            # (B, 1024, 16, 16)
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            # (B, 1024, 8, 8)
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            # (B, 1024, 4, 4)
            
            nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            # (B, 1024, 8, 8)
            nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            # (B, 1024, 16, 16)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            # (B, 512, 32, 32)
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            # (B, 512, 64, 64)
            nn.ConvTranspose2d(in_channels=512, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            # (B, 32, 128, 128)
        )

    def forward(self, x):
        x = x.squeeze(0)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.unsqueeze(0)
        return [x]