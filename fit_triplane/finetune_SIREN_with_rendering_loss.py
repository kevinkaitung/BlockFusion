from simple_raymarcher_with_shadow import *
import argparse
import numpy as np
from pathlib import Path
import sys
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from timevarying_data_helper import fibonacci_sphere, cartesian_to_spherical_coords, spherical_coords_radiance_to_normalized

def map_to_tfn_range(normalized_value, raw_data_min, raw_data_max, tfn_scalar_mapping_range_min, tfn_scalar_mapping_range_max):
    # Step 1: unnormalize back to raw data space
    raw_value = normalized_value * (raw_data_max - raw_data_min) + raw_data_min

    # Step 2: normalize into tfn range
    tfn_range = tfn_scalar_mapping_range_max - tfn_scalar_mapping_range_min
    tfn_normalized = (raw_value - tfn_scalar_mapping_range_min) / tfn_range

    # Step 3: # zero out elements outside [0, 1]
    tfn_normalized[(tfn_normalized < 0.0) | (tfn_normalized > 1.0)] = 0.0
    return tfn_normalized

def prepare_pre_calculated_sampled_points(camera, device, cfg, scene_aabb, sampler, sampler_device):
    assert device == sampler_device, "device that tensors operate on should be the same as sampler device"
    
    ray_origins, ray_directions = generate_rays(camera, device)
    H, W, _ = ray_origins.shape
    
    # --- Sample t values along each ray ---
    t_vals = torch.linspace(cfg.t_near, cfg.t_far, cfg.n_samples, device=device)
    # Perturb samples slightly (optional, helps reduce banding)
    if cfg.n_samples > 1:
        dt = (cfg.t_far - cfg.t_near) / cfg.n_samples
        noise = torch.rand(H, W, cfg.n_samples, device=device) * dt
        t_vals = t_vals.unsqueeze(0).unsqueeze(0) + noise   # (H, W, N)
    else:
        t_vals = t_vals.unsqueeze(0).unsqueeze(0).expand(H, W, -1)

    # --- World-space sample positions ---
    # origins: (H, W, 1, 3),  directions: (H, W, 1, 3),  t: (H, W, N, 1)
    pts = (ray_origins.unsqueeze(2)
        + ray_directions.unsqueeze(2) * t_vals.unsqueeze(-1))  # (H, W, N, 3)
    
    # --- Map world coords to [0,1] volume space via AABB ---
    aabb_min = scene_aabb[0].to(device)   # (3,)
    aabb_max = scene_aabb[1].to(device)   # (3,)
    
    pts_coords_norm = (pts - aabb_min) / (aabb_max - aabb_min + 1e-8)   # (H, W, N, 3)
    pts_coords_norm = pts_coords_norm.reshape(-1, 3).to(sampler_device)
    pts_values = torch.zeros([pts_coords_norm.shape[0], 1], device=sampler_device, dtype=torch.float32)
    decode(sampler, pts_coords_norm, pts_values)
    pts_coords_norm = pts_coords_norm.reshape(H, W, cfg.n_samples, 3)
    pts_values = pts_values.reshape(H, W, cfg.n_samples, 1)
    
    # -- mask: True for points inside the bounding box [0, 1]^3 --
    inside_mask = ((pts_coords_norm >= aabb_min) & (pts_coords_norm <= aabb_max)).all(dim=-1)
    # TODO: probably can be used to filter out those pixels representing the background
    pts_values[~inside_mask] = 0.0
    
    # concatenate sampled pts scalar values after sampled pts coords
    pts_coords_values = torch.cat([pts_coords_norm, pts_values], dim=-1)
    
    return pts_coords_values, inside_mask


def ray_march_with_precalculated_pts_training(
        ray_sampled_pts:    torch.Tensor,   # (n_rays, N, 4)
        inside_mask:        torch.Tensor,   # (n_rays, N)
        tfn_lut:            torch.Tensor,   # (lut_size, 4)
        cfg:                MarchConfig,
        tfn_file:           str,
        light_dir_normalized: list = [0.25, 0.25],
        net:                Any = None,
    ):
        device = ray_sampled_pts.device
        
        pts_flat = ray_sampled_pts.reshape(-1, 4)   # (n_rays*N, 4)
        inside_mask = inside_mask.flatten()     # (n_rays*N)
        
        # density_flat = torch.zeros([pts_flat.shape[0], 1], device=device)   # (n_rays*N, 1)
        # decode(self.sampler, pts_flat, density_flat)
        
        density_flat = pts_flat[:, 3:]      # (n_rays*N, 1)
        pts_coords = pts_flat[:, :3].clone()
        shadow_flat  = torch.zeros([pts_flat.shape[0], 1], device=device)   # (n_rays*N, 1)
        # C = 65536
        # for i in range(0, pts_coords.shape[0], C):
        #     shadow_flat[i:i+C] = net(pts_coords[i:i+C])
        shadow_flat = net(pts_coords)
        
        
        # zero out any outside points that decode might have affected
        # density_flat[~inside_mask] = 0.0
        shadow_flat[~inside_mask]  = 0.0
        
        # del inside_mask
        # torch.cuda.empty_cache()

        density = density_flat.reshape(-1, cfg.n_samples, 1)          # (n_rays, N, 1)
        shadow  = shadow_flat.reshape(-1, cfg.n_samples, 1)          # (n_rays, N, 1)

        # -- transfer function lookup --
        rgba    = sample_transfer_function(tfn_lut, density)    # (n_rays, N, 1, 4)
        rgba    = rgba.squeeze(2)                                    # (n_rays, N, 4)
        rgb     = rgba[..., :3]                                      # (n_rays, N, 3)
        alpha   = rgba[..., 3:]                                       # (n_rays, N, 1)

        # -- opacity correction for actual step size --
        # step_size = (cfg.t_far - cfg.t_near) / cfg.n_samples
        # alpha   = opacity_correction(alpha, step=step_size)        # (H, W, N)

        # -- shadow blending: modulate rgb by shadow coefficient --
        shadow = shadow.clamp(0.0, 1.0)             # (n_rays, N, 1)
        ambient = 1.4
        
        # should copy rgb n_batch times
        rgb = torch.lerp(rgb * ambient,
                        rgb * ambient * shadow,
                        0.9)                   # (n_rays, N, 3)
        
        alpha_c = alpha     # (n_rays, N, 1)
        
        transmittance = torch.cumprod(
            torch.cat([
                torch.ones(alpha_c.shape[0], 1, 1, device=device), # T_0 = 1 (no occlusion yet)
                1.0 - alpha_c + 1e-10                                    # (n_rays, N, 1)
            ], dim=1),
            dim=1
        )[:, :-1, :]                                                  # (n_rays, N, 1) drop the last
        
        weights = transmittance * alpha_c                             # (n_rays, N, 1)

        # -- final compositing --
        rendered_rgb = (weights * rgb).sum(dim=1)          # (n_rays, 3)
        return rendered_rgb


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', type=int, nargs=3, default=[990, 990, 1584])
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--raw_data_file_path', type=str, default="/home/kctung/Projects/BlockFusion/datasets/cranium_990x990x1584_float32.raw")
    parser.add_argument('--tfn_file_path', type=str, default="/home/kctung/Projects/BlockFusion/datasets/scene_cranium.json")
    parser.add_argument('--pretrained_SIREN_file_path', type=str, default="/home/kctung/Projects/HyperDiffusion/logs/cranium_2000_ft_w_rendering_loss_128_imp_sam/2026-03-19_05-00-12/sample_siren_8999_selected_train_set.pt.pt")
    parser.add_argument('--instance_to_finetune', type=int, default=2)
    parser.add_argument('--image_resolution', type=int, nargs=2, default=[384, 384])
    # parser.add_argument('--rendered_imgs_view_angles_file_path', type=str)
    parser.add_argument('--expname', type=str, default='test')
    parser.add_argument('--lpips_weight', type=float, default=1.0)
    
    args = parser.parse_args()
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    
    # read tfn
    tfn_file_path = args.tfn_file_path
    with open(tfn_file_path, 'r') as f:
        tfn_json = json5.load(f)
    resolution = tfn_json["dataSource"][0]["dimensions"]
    resolution = [resolution["x"], resolution["y"], resolution["z"]]
    loaded_tfn = tfn_json["view"]["volume"]["transferFunction"]
    colorControls = loaded_tfn["colorControls"]
    if "opacityControl" in loaded_tfn:
        opacityControl = loaded_tfn["opacityControl"]
        gaussianObjects = None
    elif "gaussianObjects" in loaded_tfn:
        opacityControl = None
        gaussianObjects = loaded_tfn["gaussianObjects"]

    # read value range of tfn
    tfn_scalar_mapping_range_max = tfn_json["view"]["volume"]["scalarMappingRange"]["maximum"]
    tfn_scalar_mapping_range_min = tfn_json["view"]["volume"]["scalarMappingRange"]["minimum"]

    # read raw volume just to get original min/max value
    raw_data = np.fromfile(args.raw_data_file_path, dtype=args.dtype)
    raw_data_max = raw_data.max()
    raw_data_min = raw_data.min()

    # create sampler
    resolution = args.dims
    sampler = create_sampler("structuredRegular", "cuda", dims=args.dims, dtype=args.dtype, n_channels=1, filename=args.raw_data_file_path)

    # read pretrained SIREN model and gather all instances' light directions
    pretrained_SIREN = torch.load(args.pretrained_SIREN_file_path, map_location="cpu")
    n_instances = len(pretrained_SIREN['light_dir_cartesian'])
    pretrained_SIREN_light_dirs = spherical_coords_radiance_to_normalized(cartesian_to_spherical_coords(np.array(pretrained_SIREN['light_dir_cartesian'])))
    
    # prepare and create saving directory 
    path = Path(args.pretrained_SIREN_file_path)
    save_dir = os.path.join(path.parent, f"rendering_loss_finetune_idx_{args.instance_to_finetune}_{args.expname}")
    os.makedirs(save_dir, exist_ok=True)
    pretrained_SIREN_file_path_stem = path.stem

    # prepare scene configuration
    cam = Camera(
        # for spider
        position = torch.tensor([-0.175, 0.1501, -0.17552]),
        look_at  = torch.tensor([0.4, 0.35,  0.45]),
        # up       = torch.tensor([0.0, 1.0,  0.0]),
        # position = torch.tensor([-1.091, -0.717, -1.155]),
        # look_at  = torch.tensor([-0.065, -0.437,  -0.245]),
        up       = torch.tensor([-0.758, 0.473,  0.443]),
        fov_y    = 60.0,
        width    = args.image_resolution[0],
        height   = args.image_resolution[1],
    )
    aabb = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
    cfg = MarchConfig(
        t_near    = 0.5,
        t_far     = 1.3,
        # n_samples = view_angles_file["n_samples"] if "n_samples" in view_angles_file.keys() else 1024,
        n_samples=768,
        patch_width=32,
        patch_height=32,
    )
    
    # prepare GT images
    with torch.no_grad():
        tfn_lut = build_transfer_function(colorControls, opacityControl, gaussianObjects, lut_size=1024)
        
        pts_coords_values, inside_mask = prepare_pre_calculated_sampled_points(cam, device, cfg, aabb, sampler, "cuda")
        # HACK: because some dataset's tfn doesn't fully cover raw data's value range
        # need additional process
        pts_coords_values[..., 3] = map_to_tfn_range(pts_coords_values[..., 3], raw_data_min, raw_data_max, tfn_scalar_mapping_range_min, tfn_scalar_mapping_range_max)
        
        light_dir_normalized = pretrained_SIREN_light_dirs[args.instance_to_finetune].tolist()
        
        # reshape to align with the input shape expected in ray_march_with_precalculated_pts
        GT_image = ray_march_with_precalculated_pts(pts_coords_values.reshape(-1, cfg.n_samples, 4), 
                                                    inside_mask.reshape(-1, cfg.n_samples), sampler, 
                                                    tfn_lut, cfg, tfn_file_path, light_dir_normalized)
        # should reshape result back to have H, W
        GT_image = GT_image.reshape(args.image_resolution[0], args.image_resolution[1], -1)
        
        # save GT images
        plt.figure(figsize=(7, 7))
        data = GT_image.detach().cpu().numpy()
        im = plt.imshow(data)
        plt.title(f"Full render with shadow")
        plt.savefig(os.path.join(save_dir,f"GT_image_for_finetuning.png"))
        plt.close()

        # import pdb; pdb.set_trace()

    nets = [NeurCompNet(n_input_dims=3, 
                    n_output_dims=1, bias=False, 
                    n_hidden_layers=4, 
                    n_neurons=128, is_residual=True).cuda() for _ in range(n_instances)]
    nets = torch.nn.ModuleList(nets)
    nets.load_state_dict(pretrained_SIREN['net_state_dict'])
    nets.train()
    net_0 = nets[args.instance_to_finetune]

    # prepare training modules (e.g., optimizer etc.)
    tensorboard_writer = SummaryWriter(log_dir=save_dir)
    optimizer = torch.optim.Adam(net_0.parameters(), lr=5e-5)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', reduction='mean').to(device)
    lpips_weight = args.lpips_weight

    full_images = torch.zeros_like(GT_image)
    
    iteration = 0
    epochs = 600
    for epoch in tqdm(range(epochs)):

        # patchify the images for rendering
        # iterate over non-overlapping patches
        for y in range(0, cam.height, cfg.patch_height):
            for x in range(0, cam.width, cfg.patch_width):
                
                # slice a patch of rays
                pts_coords_values_patch = pts_coords_values[y:y+cfg.patch_height, x:x+cfg.patch_width]   # (ph, pw, 3)
                inside_mask_patch = inside_mask[y:y+cfg.patch_height, x:x+cfg.patch_width]   # (ph, pw, 3)
                GT_image_patch = GT_image[y:y+cfg.patch_height, x:x+cfg.patch_width]   # (ph, pw, 3)

                result = ray_march_with_precalculated_pts_training(
                    ray_sampled_pts =pts_coords_values_patch,
                    inside_mask     =inside_mask_patch,
                    tfn_lut         = tfn_lut,
                    cfg             = cfg,
                    tfn_file        = None,
                    light_dir_normalized=None,
                    net             = net_0
                )
                result = result.reshape(cfg.patch_height, cfg.patch_width, -1)
                # import pdb; pdb.set_trace()
                # lpips expect the channel dim to be the first
                # and expect the value range to be -1~1
                lpips_loss = lpips(result.permute(2, 0, 1).unsqueeze(0) * 2 - 1, GT_image_patch.permute(2, 0, 1).unsqueeze(0) * 2 - 1)
                mse_loss = torch.nn.functional.mse_loss(result, GT_image_patch)
                loss = lpips_weight * lpips_loss + mse_loss
                optimizer.zero_grad()
                loss.backward()          # graph freed here
                optimizer.step()
                print(f"epoch: {epoch}, iteration: {iteration}, loss: {loss.item()}, mse_loss: {mse_loss.item()}, lpips_loss: {lpips_loss.item()}")
                tensorboard_writer.add_scalar("loss", loss.item(), iteration)
                tensorboard_writer.add_scalar("mse_loss", mse_loss.item(), iteration)
                tensorboard_writer.add_scalar("lpips_loss", lpips_loss.item(), iteration)
                iteration += 1
                
                if epoch == epochs - 1 or epoch % 25 == 0 or epoch < 10:
                    full_images[y:y+cfg.patch_height, x:x+cfg.patch_width] = result
                    
         
        if epoch == epochs - 1 or epoch % 25 == 0 or epoch < 10:
            data = full_images
            plt.figure(figsize=(7, 7))
            # Convert to numpy if tensor
            if torch.is_tensor(data):
                data = data.detach().cpu().numpy()
            # Plot with colorbar
            if data.shape[-1] == 3:
                im = plt.imshow(data)
            else:
                im = plt.imshow(data, cmap='viridis')
            # cbar = plt.colorbar()
            plt.title(f"Full render with shadow during finetuning (epoch: {epoch})")
            plt.savefig(os.path.join(save_dir,f"render_during_finetune_epoch{epoch}.png"))
            plt.close()

    print(f"max memory allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"max memory reserved: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

    finetune_result_to_save = dict()
    # HACK: only one sample now
    finetune_result_to_save["finetuned_image"] = full_images
    # finetune_result_to_save["light_dir_spherical_normalized"] = light_dir_normalized
    finetune_result_to_save["light_dir_cartesian"] = pretrained_SIREN['light_dir_cartesian']
    finetune_result_to_save['net_state_dict'] = nets.state_dict()
    # save image tensor as pt file
    torch.save(finetune_result_to_save, os.path.join(save_dir, "finetuned_image.pt"))