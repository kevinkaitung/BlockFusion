import torch
from diffusers import DDPMScheduler
# from torchvision.utils import save_image
import os
import argparse

# --- Step 1: Load pretrained model and scheduler ---
# Assuming you already have a trained U-Net model
# Replace this with your custom model import if needed
from diffusers import Conv3DAwareUNet, Conv3DAwareUNet2DConditionModel  # or any U-Net class you used
from diffusion_training import FourierEmbedder
from timevarying_data_helper import ShadowVolumesMetaDataset, ShadowLightingDirectionsDataset, spherical_to_cartesian_coords

import numpy as np

num_freqs = 64
# number of freqs * 3 coordinates * 2 (sin and cos)
embed_dim = num_freqs * 3 * 2

import matplotlib.pyplot as plt
def plot_single_channel(data, path_to_save):
    plt.imshow(data.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title("2D Tensor Visualization")
    plt.savefig(path_to_save)
    plt.close()
    
def plot_histogram(data, path_to_save):
    # Plot histogram
    plt.hist(data.cpu().numpy().flatten(), bins=100, density=True)
    plt.title("Tensor Value Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig(path_to_save)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="UNet Diffusion Inference for latent triplanes")
    parser.add_argument("--expdir", type=str, default="logs/UNet_diffusion_exp/20250624-162156", help="Checkpoint Directory to load the model from")
    parser.add_argument("--model_file_name", type=str, default="vae_model_epoch_9999.ckpt", help="Model file name")
    parser.add_argument("--latent_triplanes_file_path", type=str, default="logs/triplane_AE_model_a/20250619-000758", help="Directory where latent triplanes are stored")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size of generated samples")
    # load pretrained triplane just to get the lighting direction of each instance
    # TODO: incorporate the lighting direction info into latent triplanes file
    parser.add_argument("--pretrained_triplane_file_path", type=str, default=None, help="File Path to Pretrained Triplanes Model")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # plane_shape = [3, 32, 128, 128]
    plane_shape = [3, 4, 32, 32]

    # # Load model and put it in eval mode
    # model = Conv3DAwareUNet(
    #         sample_size=plane_shape[2:],
    #         in_channels=plane_shape[1],
    #         out_channels=plane_shape[1],
    #         # block_out_channels=(128, 256, 384, 512)
    #         block_out_channels=(128, 256, 512, 1024),
    #         layers_per_block=3
    #         # rest of the arguments uses default values
    #     ).cuda()
    
    model = Conv3DAwareUNet2DConditionModel(
        sample_size=plane_shape[2:],
        in_channels=plane_shape[1],
        out_channels=plane_shape[1],
        down_block_types=("DownBlock2D", "SimpleCrossAttnDownBlock2D", "SimpleCrossAttnDownBlock2D", "SimpleCrossAttnDownBlock2D"),
        up_block_types=("SimpleCrossAttnUpBlock2D", "SimpleCrossAttnUpBlock2D", "SimpleCrossAttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(128, 256, 512, 1024),
        layers_per_block=2,
        cross_attention_dim=embed_dim,
        # rest of the arguments uses default values
    ).cuda()
    
    model.load_state_dict(torch.load(os.path.join(args.expdir, args.model_file_name))["model_state_dict"])
    model = model.cuda().eval()
    # model.eval()
    
    positional_embedder = FourierEmbedder(num_freqs=num_freqs).cuda()

    # pretrained_triplane_model = torch.load("fit_triplane/ch_32_saved_model.ckpt")
    latent_triplanes = torch.load(args.latent_triplanes_file_path)
    latent_triplanes = latent_triplanes["weights_latent_space"]
    
    # shadow_meta_dataset = ShadowVolumesMetaDataset(
    #     raw_data_dir="/home/kctung/Ring1Light",
    #     raw_data_filename_prefix="shadow",
    #     file_ext="json",
    #     n_instances=latent_triplanes.shape[0],
    # )
    shadow_meta_dataset = ShadowLightingDirectionsDataset(
        lighting_dirs=torch.load(args.pretrained_triplane_file_path)["light_dir_cartesian"]
    )

    # Setup scheduler (should match the one used in training)
    scheduler = DDPMScheduler(num_train_timesteps=257)
    scheduler.set_timesteps(200)  # Can reduce for faster inference

    # --- Step 2: Generate Gaussian noise as initial input ---
    # image_size = (32, 128, 128 * 3)  # Replace with your output shape
    image_size = (plane_shape[1], plane_shape[2], plane_shape[3] * plane_shape[0])
    noise = torch.randn((args.batch_size, *image_size)).cuda()

    pos_embed = positional_embedder(shadow_meta_dataset[100:100+args.batch_size])
    # This is for evaluting the diffusion-generated shadow under randomly-generated lighting direction
    # randomly_generated_lightdir = np.random.rand(1, 2)
    # pos_embed = positional_embedder(torch.tensor(spherical_to_cartesian_coords(randomly_generated_lightdir)).cuda().float())
    # make the shape to be [batch_size, sequence_length (currently one for representing one light), feature_dim]
    pos_embed = pos_embed[0].unsqueeze(1)
    # --- Step 3: Perform reverse diffusion process ---
    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_input = noise
            model_output = model(noise_input, t, pos_embed).sample
            noise = scheduler.step(model_output, t, noise)["prev_sample"]
    # only permute for raw triplanes
    # noise = noise.view(1, plane_shape[1], plane_shape[2], 3, plane_shape[3]).permute(0, 3, 1, 2, 4)
    # denormalize from -1~1 to its original (latent_triplanes) value range
    noise = (noise - (-1)) / 2.0 * (latent_triplanes.max() - latent_triplanes.min()) + latent_triplanes.min()
    torch.save({"weights_latent_space": noise}, os.path.join(args.expdir, "diffusion_latent_triplanes.pt"))

    # This is for raw triplane diffusion inference
    # # TODO: replace len(pretrained_triplane_model['triplane_state_dict'])//2 with actual n_timesteps
    # for i in range(len(pretrained_triplane_model['triplane_state_dict'])//2):
    #     # since we only generate one set of triplane to see the overall reconstruction quality,
    #     # copy that for every timestep
    #     pretrained_triplane_model['triplane_state_dict'][f"{i}.triplane"] = noise
    # torch.save(pretrained_triplane_model, "Diffusion_Reconstructed_triplane.pt")
    # # import pdb; pdb.set_trace()
    # # --- Step 4: Save or view result ---
    # # save_image(noise, "generated_samples.png", nrow=2, normalize=True)
