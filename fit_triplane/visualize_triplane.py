import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_single_channel(data, title="Single Channel Plot", cmap='viridis', save_path=None):
    """
    Plot a single channel 2D image
    Args:
        data: 2D numpy array or tensor
        title: plot title
        cmap: colormap (viridis, gray, jet, etc)
        save_path: if provided, saves the plot to this path
    """
    plt.figure(figsize=(8, 8))
    
    # Convert to numpy if tensor
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    
    # Plot with colorbar
    plt.imshow(data, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":

    vae_recon_model = torch.load("../VAE_Reconstructed_triplane_ch_32.pt")
    triplane_model = torch.load("ch_32_saved_model.ckpt")


    idx = 75

    vae_recon_triplane = [vae_recon_model['triplane_state_dict'][f"{idx}.triplane"][0][i] for i in range(3)]
    original_triplane = [triplane_model['triplane_state_dict'][f"{idx}.triplane"][0][i] for i in range(3)]

    for feature_idx in range(16, 17):
        # Example usage for your triplane visualization:
        for i, (vae_plane, orig_plane) in enumerate(zip(vae_recon_triplane, original_triplane)):
            # Plot VAE reconstructed plane
            plot_single_channel(
                vae_plane[feature_idx], 
                title=f"VAE Reconstructed Plane {i}, Feature {feature_idx}",
                save_path=f"vae_plane_{i}_feature_{feature_idx}.png"
            )
            
            # Plot original plane
            plot_single_channel(
                orig_plane[feature_idx],
                title=f"Original Plane {i}, Feature {feature_idx}",
                save_path=f"orig_plane_{i}_feature_{feature_idx}.png"
            )