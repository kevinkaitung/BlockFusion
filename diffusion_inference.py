import torch
from diffusers import DDPMScheduler
from torchvision.utils import save_image

# --- Step 1: Load pretrained model and scheduler ---
# Assuming you already have a trained U-Net model
# Replace this with your custom model import if needed
from diffusers import Conv3DAwareUNet  # or any U-Net class you used

plane_shape = [3, 64, 128, 128]

# Load model and put it in eval mode
model = Conv3DAwareUNet(
        # TODO: check should I pass [H, W * 3 (triplanes)] or just pass [H, W]
        # but seems like they don't use this parameter, it should be fine
        sample_size=plane_shape[2:],
        in_channels=plane_shape[1],
        out_channels=plane_shape[1],
        block_out_channels=(128, 256, 384, 512)
        # rest of the arguments uses default values
    ).cuda()
# TODO: should receive arguments to specify the location of pretrained model and other arguments
model.load_state_dict(torch.load("logs/UNet_diffusion_exp/20250605-004149/Diffusion_UNet_model_epoch_199.ckpt")["model_state_dict"])
model.cuda()
model.eval()

pretrained_triplane_model = torch.load("fit_triplane/ch_64_saved_model.ckpt")

# Setup scheduler (should match the one used in training)
scheduler = DDPMScheduler(num_train_timesteps=257)
scheduler.set_timesteps(200)  # Can reduce for faster inference

# --- Step 2: Generate Gaussian noise as initial input ---
batch_size = 1
image_size = (64, 128, 128 * 3)  # Replace with your output shape
noise = torch.randn((batch_size, *image_size)).cuda()

# --- Step 3: Perform reverse diffusion process ---
with torch.no_grad():
    for t in scheduler.timesteps:
        noise_input = noise
        model_output = model(noise_input, t).sample
        # import pdb; pdb.set_trace()
        noise = scheduler.step(model_output, t, noise)["prev_sample"]
noise = noise.view(1, plane_shape[1], plane_shape[2], 3, plane_shape[3]).permute(0, 3, 1, 2, 4)

# TODO: replace len(pretrained_triplane_model['triplane_state_dict'])//2 with actual n_timesteps
for i in range(len(pretrained_triplane_model['triplane_state_dict'])//2):
    # since we only generate one set of triplane to see the overall reconstruction quality,
    # copy that for every timestep
    pretrained_triplane_model['triplane_state_dict'][f"{i}.triplane"] = noise
torch.save(pretrained_triplane_model, "Diffusion_Reconstructed_triplane.pt")
# import pdb; pdb.set_trace()
# --- Step 4: Save or view result ---
# save_image(noise, "generated_samples.png", nrow=2, normalize=True)
