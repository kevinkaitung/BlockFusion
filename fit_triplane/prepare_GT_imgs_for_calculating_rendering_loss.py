from simple_raymarcher_with_shadow import *
import argparse
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from dataclasses import asdict

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from timevarying_data_helper import fibonacci_sphere, cartesian_to_spherical_coords, spherical_coords_radiance_to_normalized

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--raw_data_file_path', type=str, default="/media/data/qadwu/volume/vortices/vorts1.data")
    parser.add_argument('--tfn_file_path', type=str, default="/home/kctung/Projects/instant-vnr-pytorch/bindings/ovr/data/configs/vorts_shadow.json")
    parser.add_argument('--pretrained_SIREN_file_path', type=str)
    parser.add_argument('--image_resolution', type=int, nargs=2, default=[128, 128])
    parser.add_argument('--rendered_imgs_view_angles_file_path', type=str)
    
    args = parser.parse_args()
    
    # read tfn
    tfn_file_path = args.tfn_file_path
    with open(tfn_file_path, 'r') as f:
        tfn_json = json5.load(f)
    resolution = tfn_json["dataSource"][0]["dimensions"]
    resolution = [resolution["x"], resolution["y"], resolution["z"]]
    colorControls = tfn_json["view"]["volume"]["transferFunction"]["colorControls"]
    opacityControl = tfn_json["view"]["volume"]["transferFunction"]["opacityControl"]

    # create sampler
    resolution = args.dims
    sampler = create_sampler("structuredRegular", "cuda", dims=args.dims, dtype=args.dtype, n_channels=1, filename=args.raw_data_file_path)

    # read pretrained SIREN model and gather all instances' light directions
    pretrained_SIREN = torch.load(args.pretrained_SIREN_file_path, map_location="cpu")
    n_instances = len(pretrained_SIREN['light_dir_cartesian'])
    pretrained_SIREN_light_dirs = spherical_coords_radiance_to_normalized(cartesian_to_spherical_coords(np.array(pretrained_SIREN['light_dir_cartesian'])))
    
    # prepare saving directory
    path = Path(args.pretrained_SIREN_file_path)
    save_dir = path.parent
    pretrained_SIREN_file_path_stem = path.stem
    
    
    # generate camera positions from fibonacci sphere (which is numpy array)
    # fibonacci_points = fibonacci_sphere(32)
    # fibonacci_points = fibonacci_points.astype(np.float32)
    # pre-select some of the points (more front-facing points)
    # for spider dataset
    # fibonacci_points = [
    #    [-0.06371014, -0.96875   ,  0.23971745],
    #    [-0.19591164, -0.90625   , -0.37460588],
    #    [ 0.50468982, -0.84375   ,  0.18268586],
    #    [-0.57631327, -0.78125   ,  0.23981546],
    #    [ 0.29289361, -0.71875   , -0.63056465],
    #    [ 0.22787057, -0.65625   ,  0.71931282],
    #    [-0.69733713, -0.59375   , -0.40147461],
    #    [ 0.82693345, -0.53125   , -0.18426749],
    # #    [ 0.62790148, -0.46875   ,  0.62129958], 8 
    #    [-0.91308541, -0.40625   , -0.03515644],
    #    [ 0.71632715, -0.34375   , -0.60721607],
    # #    [-0.12061461, -0.28125   ,  0.95202445], 11
    #    [-0.56346964, -0.21875   , -0.79664949],
    #    [ 0.96527941, -0.15625   ,  0.2093361 ],
    #    [-0.85997656, -0.09375   ,  0.50164853],
    #    [ 0.29642255, -0.03125   , -0.9545455 ],
    # #    [ 0.42621346,  0.03125   ,  0.9040827 ], 16
    #    [-0.92135181,  0.09375   , -0.37725559],
    #    [ 0.92681617,  0.15625   , -0.34146409],
    # #    [-0.44727822,  0.21875   ,  0.86723159], 19
    #    [-0.25176145,  0.28125   , -0.92602085],
    # #    [ 0.79376894,  0.34375   ,  0.5017637 ], 21
    #    [-0.89933717,  0.40625   ,  0.16172074],
    #    [ 0.53545579,  0.46875   , -0.70253864],
    # #    [ 0.07646978,  0.53125   ,  0.84375696], 24
    # #    [-0.59486788,  0.59375   , -0.54184236], 25
    # #    [ 0.75454047,  0.65625   , -0.0021472 ],
    # #    [-0.51133088,  0.71875   ,  0.4711042 ],
    # #    [ 0.05280298,  0.78125   , -0.62198093], 28
    # #    [ 0.32778208,  0.84375   ,  0.42502334],
    # #    [-0.41648776,  0.90625   , -0.07244915], 30
    #    [ 0.20890468,  0.96875   , -0.13372461]
    # ]
    # # for mechhand dataset
    # fibonacci_points = [
    #     [-0.48411712, -0.875,      -0.00236781],
    #     [ 0.5781805,  -0.625,      -0.52448285],
    #     [ 0.23627819, -0.375,       0.8964082 ],
    #     [-0.83452296, -0.125,      -0.5366064 ],
    #     [ 0.9778237,   0.125,      -0.16803534],
    #     [-0.5676294,   0.375,       0.7329201 ],
    #     [-0.06444251,  0.625,      -0.77796024],
    #     # [ 0.35537347,  0.875,       0.32876238],
    # ]
    # for spider dataset
    # fibonacci_points = [
    #     [-0.48411712, -0.875,      -0.00236781],
    #     [ 0.5781805,  -0.625,      -0.52448285],
    #     [ 0.23627819, -0.375,       0.8964082 ],
    #     [-0.83452296, -0.125,      -0.5366064 ],
    #     [ 0.9778237,   0.125,      -0.16803534],
    #     [-0.69733713, -0.59375   , -0.40147461],
    #     [-0.06444251,  0.625,      -0.77796024],
    #     # [ 0.35537347,  0.875,       0.32876238],
    # ]
    with open(args.rendered_imgs_view_angles_file_path, 'r') as f:
        view_angles_file = json5.load(f)
    fibonacci_points = view_angles_file["fibonacci_points"]
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
        n_samples = 1024,
        # no use of patch
        patch_width=16,
        patch_height=16,
    )
    
    with torch.no_grad():
        
        # need to translate by the center point (look at)
        # because fibonacci sphere was generated based on the origin as sphere center
        fibonacci_points = torch.tensor(fibonacci_points)
        fibonacci_points = fibonacci_points + cam.look_at
        
        tfn_lut = build_transfer_function(colorControls, opacityControl, lut_size=1024)
        
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
            # pts_coords_values_group.append(pts_coords_values.cpu())
            # inside_mask_group.append(inside_mask.cpu())
            pts_coords_values_group.append(pts_coords_values)
            inside_mask_group.append(inside_mask)
        
        pre_cal_GT_images = []
        # iterate through all instances
        for instance_idx in tqdm(range(n_instances)):
            print(f"Processing instance idx: {instance_idx}")
            
            light_dir_normalized = pretrained_SIREN_light_dirs[instance_idx].tolist()
            GT_images_this_instance = []
            
            # iterate through all camera position
            for batch_idx, (pts_coords_values, inside_mask) in enumerate(zip(pts_coords_values_group, inside_mask_group)):
                
                # reshape to align with the input shape expected in ray_march_with_precalculated_pts
                result = ray_march_with_precalculated_pts(pts_coords_values.reshape(-1, cfg.n_samples, 4), 
                                                          inside_mask.reshape(-1, cfg.n_samples), sampler, 
                                                          tfn_lut, cfg, tfn_file_path, light_dir_normalized)
                # should reshape result back to have H, W
                result = result.reshape(args.image_resolution[0], args.image_resolution[1], -1)
                                
                # save GT images for some instances
                if instance_idx == 0 or instance_idx == 100 or instance_idx == 200:
                    plt.figure(figsize=(7, 7))
                    data = result.detach().cpu().numpy()
                    im = plt.imshow(data)
                    plt.title(f"Full render with shadow from camera pos: {cam_position}")
                    plt.savefig(os.path.join(save_dir,f"GT_{args.image_resolution[0]}x{args.image_resolution[1]}_image_ins_{instance_idx}_cam_{batch_idx}.png"))
                    plt.close()
                
                GT_images_this_instance.append(result.cpu())
                
            pre_cal_GT_images.append(torch.stack(GT_images_this_instance))
    
    # save pre-calculated GT images
    pretrained_SIREN["pre_cal_GT_images"] = pre_cal_GT_images
    
    # save all camera and marching configs
    # TODO: integrate with the above code
    camera_configs = []
    aabb_configs = []
    march_configs = []
    for batch_idx, (cam_position, t_near_far) in enumerate(zip(fibonacci_points, ts_near_far)):
        camera_configs.append(asdict(Camera(
            position = cam_position,
            look_at  = cam.look_at,
            up       = cam.up,
            fov_y    = cam.fov_y,
            width    = cam.width,
            height   = cam.height,
        )))
        aabb_configs.append(torch.tensor([[0., 0., 0.], [1., 1., 1.]]))
        march_configs.append(asdict(MarchConfig(
            t_near    = t_near_far[0],
            t_far     = t_near_far[1],
            n_samples = cfg.n_samples,
            # no use of patch
            patch_width=cfg.patch_width,
            patch_height=cfg.patch_height,
        )))
    
    pretrained_SIREN["camera_configs"] = camera_configs
    pretrained_SIREN["aabb_configs"] = aabb_configs
    pretrained_SIREN["march_configs"] = march_configs
    # move those tensors back to cpu for consistency as GT images
    pretrained_SIREN["pts_coords_values_group"] = [pts_coords_values.cpu() for pts_coords_values in pts_coords_values_group]
    pretrained_SIREN["inside_mask_group"] = [inside_mask.cpu() for inside_mask in inside_mask_group]
    
    torch.save(pretrained_SIREN, os.path.join(save_dir, f"{pretrained_SIREN_file_path_stem}_w_GT_{args.image_resolution[0]}x{args.image_resolution[1]}_imgs.pt"))
    
    print(f"max memory allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"max memory reserved: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")
        