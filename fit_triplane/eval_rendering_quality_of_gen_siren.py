from simple_raymarcher_with_shadow import *
import argparse
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from dataclasses import asdict
from networks import NeurCompNet

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

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

def compare_images(gt_img, gen_img):

    assert gt_img.shape == gen_img.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_img = gt_img.to(device)
    gen_img = gen_img.to(device)

    # -------------------------
    # PSNR (manual)
    # -------------------------
    def compute_psnr(x, y):
        mse = torch.mean((x - y) ** 2)
        return -10 * torch.log10(mse)

    # filter out background for psnr eval
    fg_mask = (gt_img > 0.01).any(dim=1).squeeze(0)
    psnr_val = compute_psnr(gt_img.permute(2,3,1,0).squeeze(-1)[fg_mask], gen_img.permute(2,3,1,0).squeeze(-1)[fg_mask])
    
    # -------------------------
    # MS-SSIM (TorchMetrics)
    # -------------------------
    msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    msssim_val = msssim_metric(gt_img, gen_img)

    # -------------------------
    # LPIPS (TorchMetrics)
    # -------------------------
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    # LPIPS expects [-1,1]
    gt_img_lp = gt_img * 2 - 1
    gen_img_lp = gen_img * 2 - 1

    lpips_val = lpips_metric(gt_img_lp, gen_img_lp)

    # -------------------------
    # Print
    # -------------------------
    print(f"PSNR:    {psnr_val.item():.4f} dB")
    print(f"MS-SSIM: {msssim_val.item():.6f}")
    print(f"LPIPS:   {lpips_val.item():.6f}")
    return psnr_val.item(), msssim_val.item(), lpips_val.item()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', type=int, nargs=3, default=[592, 413, 956])
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--raw_data_file_path', type=str, default="/home/kctung/Projects/BlockFusion/datasets/zebrafish_592x413x956_float32.raw")
    parser.add_argument('--tfn_file_path', type=str, default="/home/kctung/Projects/BlockFusion/datasets/scene_zebrafish.json")
    parser.add_argument('--SIREN_file_path', type=str)
    parser.add_argument('--image_resolution', type=int, nargs=2, default=[384, 384])
    parser.add_argument('--rendered_imgs_view_angles_file_path', type=str, default="/home/kctung/Projects/BlockFusion/fit_triplane/selected_view_angles/zebrafish_view_angles_test_set.json")
    
    args = parser.parse_args()
    
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
    SIREN_model = torch.load(args.SIREN_file_path, map_location="cpu")
    n_instances = len(SIREN_model['light_dir_cartesian'])
    SIREN_model_light_dirs = spherical_coords_radiance_to_normalized(cartesian_to_spherical_coords(np.array(SIREN_model['light_dir_cartesian'])))
    
    # prepare saving directory
    path = Path(args.SIREN_file_path)
    save_dir = path.parent
    SIREN_file_path_stem = path.stem
    # create a directory for saving images
    image_save_dir = os.path.join(save_dir, "eval_images")
    os.makedirs(image_save_dir, exist_ok=True)
    
    # load optimized SIREN model
    nets = [
        NeurCompNet(n_input_dims=3, n_output_dims=1, bias=False, n_hidden_layers=4, n_neurons=128, is_residual=True).cuda()
            for _ in range(n_instances)]
    nets = torch.nn.ModuleList(nets)
    nets.load_state_dict(SIREN_model['net_state_dict'])
    
    # generate camera positions from fibonacci sphere (which is numpy array)
    # fibonacci_points = fibonacci_sphere(32)
    # fibonacci_points = fibonacci_points.astype(np.float32)
    with open(args.rendered_imgs_view_angles_file_path, 'r') as f:
        view_angles_file = json5.load(f)
    fibonacci_points = view_angles_file["fibonacci_points"]
    if "ts_near_far" in view_angles_file.keys():
        ts_near_far = view_angles_file["ts_near_far"]
    else:
        ts_near_far = [
            [0.3, 1.5] for _ in range(len(fibonacci_points))
        ]
    # # permute indices from spider for mechhand
    # old selection (leave here just for record)
    # perm_indices = [12, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19]
    # fibonacci_points = [fibonacci_points[perm_idx] for perm_idx in perm_indices]

    # NOTE: used when using pytorch impl. for sampling
    # # ---- read raw file ----
    # volume_np = np.fromfile(args.raw_data_file_path, dtype=np.float32)
    # # reshape to 3D
    # volume_np = volume_np.reshape((resolution[2], resolution[1], resolution[0]))
    # # convert to torch tensor
    # volume_tensor = torch.from_numpy(volume_np)
    # # normalize volume to 0~1 with in-place operations to save memory usage            
    # volume_min = volume_tensor.min()
    # volume_max = volume_tensor.max()
    # volume_tensor.sub_(volume_min)
    # volume_tensor.div_(volume_max - volume_min)
    # # create additional dimension for channel (channel dim in this case is 1)
    # volume_tensor = volume_tensor.unsqueeze(-1) # (D, H, W, C)
    # volume_tensor = volume_tensor.to("cuda")

    # prepare scene configuration
    cam = Camera(
        # for spider
        position = torch.tensor([-0.085, -0.31, -0.012]),
        look_at  = torch.tensor([0.5, 0.5,  0.5]),
        up       = torch.tensor([0.0, 1.0,  0.0]),
        fov_y    = 60.0,
        width    = args.image_resolution[0],
        height   = args.image_resolution[1],
    )
    aabb = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
    cfg = MarchConfig(
        t_near    = 0.0001,
        t_far     = 2.0,
        n_samples = view_angles_file["n_samples"] if "n_samples" in view_angles_file.keys() else 1024,
        # no use of patch
        patch_width=16,
        patch_height=16,
    )
    
    with torch.no_grad():
        
        # need to translate by the center point (look at)
        # because fibonacci sphere was generated based on the origin as sphere center
        fibonacci_points = torch.tensor(fibonacci_points)
        fibonacci_points = fibonacci_points + cam.look_at
        
        tfn_lut = build_transfer_function(colorControls, opacityControl, gaussianObjects, lut_size=1024)
        
        pts_coords_values_group = []
        inside_mask_group = []
        # iterate through all camera position
        for batch_idx, (cam_position, t_near_far) in enumerate(zip(fibonacci_points, ts_near_far)):
            print(f"Processing batch idx: {batch_idx} / camera position: {cam_position} / t_near_far: {t_near_far}")
            
            # set to corresponding camera and config
            cam.position = cam_position
            cfg.t_near = t_near_far[0]
            cfg.t_far = t_near_far[1]
            
            pts_coords_values, inside_mask = prepare_pre_calculated_sampled_points(cam, "cuda", cfg, aabb, sampler, "cuda")
            # HACK: because some dataset's tfn doesn't fully cover raw data's value range
            # need additional process
            pts_coords_values[..., 3] = map_to_tfn_range(pts_coords_values[..., 3], raw_data_min, raw_data_max, tfn_scalar_mapping_range_min, tfn_scalar_mapping_range_max)
            
            # pts_coords_values_group.append(pts_coords_values.cpu())
            # inside_mask_group.append(inside_mask.cpu())
            pts_coords_values_group.append(pts_coords_values)
            inside_mask_group.append(inside_mask)
        
        psnr_all_instances = []
        msssim_all_instances = []
        lpips_all_instances = []
        # iterate through all instances
        for instance_idx in tqdm(range(n_instances)):
            print(f"Processing instance idx: {instance_idx}")
            
            light_dir_normalized = SIREN_model_light_dirs[instance_idx].tolist()
            
            psnr_this_instance = []
            msssim_this_instance = []
            lpips_this_instance = []
            
            # iterate through all camera position
            for batch_idx, (pts_coords_values, inside_mask) in enumerate(zip(pts_coords_values_group, inside_mask_group)):
                
                # reshape to align with the input shape expected in ray_march_with_precalculated_pts
                result_GT = ray_march_with_precalculated_pts(pts_coords_values.reshape(-1, cfg.n_samples, 4), 
                                                          inside_mask.reshape(-1, cfg.n_samples), sampler, 
                                                          tfn_lut, cfg, tfn_file_path, light_dir_normalized)
                result_gen = ray_march_with_precalculated_pts(pts_coords_values.reshape(-1, cfg.n_samples, 4), 
                                                          inside_mask.reshape(-1, cfg.n_samples), sampler, 
                                                          tfn_lut, cfg, tfn_file_path, light_dir_normalized, nets[instance_idx])
                
                # should reshape result back to have H, W
                result_GT = result_GT.reshape(args.image_resolution[0], args.image_resolution[1], -1)
                result_gen = result_gen.reshape(args.image_resolution[0], args.image_resolution[1], -1)
                                
                psnr, msssim, lpips = compare_images(result_GT.permute(2, 0, 1).unsqueeze(0), result_gen.permute(2, 0, 1).unsqueeze(0))
                
                # save GT images for some instances
                if instance_idx % 20 == 0:
                    plt.figure(figsize=(7, 7))
                    data = result_GT.detach().cpu().numpy()
                    im = plt.imshow(data)
                    plt.title(f"Full render with shadow (GT) from camera pos: {cam_position}")
                    plt.savefig(os.path.join(image_save_dir,f"GT_{args.image_resolution[0]}x{args.image_resolution[1]}_image_ins_{instance_idx}_cam_{batch_idx}.png"))
                    plt.close()
                
                    plt.figure(figsize=(7, 7))
                    data = result_gen.detach().cpu().numpy()
                    im = plt.imshow(data)
                    plt.title(f"Full render with shadow (generated) from camera pos: {cam_position}\nPSNR: {psnr:.4f} dB ms-ssim: {msssim:.6f} lpips: {lpips:.6f}")
                    plt.savefig(os.path.join(image_save_dir,f"Gen_{args.image_resolution[0]}x{args.image_resolution[1]}_image_ins_{instance_idx}_cam_{batch_idx}.png"))
                    plt.close()
                    
                
                psnr_this_instance.append(psnr)
                msssim_this_instance.append(msssim)
                lpips_this_instance.append(lpips)
                
            psnr_all_instances.append(torch.tensor(psnr_this_instance).mean().item())
            msssim_all_instances.append(torch.tensor(msssim_this_instance).mean().item())
            lpips_all_instances.append(torch.tensor(lpips_this_instance).mean().item())
    
    data_to_store = {
        "PSNR_recon_quality": psnr_all_instances,
        "msssim": msssim_all_instances,
        "lpips": lpips_all_instances,
        "light_dir_cartesian": SIREN_model['light_dir_cartesian']
    }
    
    with open(os.path.join(save_dir, f"{SIREN_file_path_stem}_img_eval_metrics.json"), "w") as f:
        import json
        json.dump(data_to_store, f, indent=4)
    
    print(f"max memory allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"max memory reserved: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")
        