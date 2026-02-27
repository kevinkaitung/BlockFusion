"""
PyTorch Ray Marcher with:
  - Perspective projection (non-parallel rays per pixel)
  - Precomputed voxel shadow coefficients (no Phong gradient needed)
  - Volume rendering via alpha compositing
  - Trilinear interpolation of volume/shadow data
"""
from simple_raymarcher_with_shadow import *
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def patchify_render_training(
    camera:       Camera,
    sampler:      Any,
    tfn_lut:      torch.Tensor,   # (lut_size, 4)
    scene_aabb:   torch.Tensor,    # (2, 3) world bounding box
    cfg:          MarchConfig,
    device:       torch.device,
    GT_image:     torch.Tensor,
    nets: Any,
    save_dir: str
) -> dict:
    """
    End-to-end render.  Loads placeholder volumes, generates rays, ray-marches.
    """
    
    # ray origins and directions might not need gradients
    with torch.no_grad():
        ray_origins, ray_directions = generate_rays(camera, device)
    
        patch_height = cfg.patch_height
        patch_width = cfg.patch_width

        GT_image = GT_image.to(device)
        full_images = torch.zeros_like(GT_image)
    
    # prepare training modules (e.g., optimizer etc.)
    tensorboard_writer = SummaryWriter(log_dir=save_dir)
    optimizer = torch.optim.Adam(nets.parameters(), lr=5e-5)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', reduction='mean').to(device)

    iteration = 0
    epochs = 1000
    for epoch in tqdm(range(epochs)):

        # patchify the images for rendering
        # iterate over non-overlapping patches
        for y in range(0, cam.height, patch_height):
            for x in range(0, cam.width, patch_width):
                
                # slice a patch of rays
                ro = ray_origins[y:y+patch_height, x:x+patch_width]   # (ph, pw, 3)
                rd = ray_directions[y:y+patch_height, x:x+patch_width]   # (ph, pw, 3)
                gt = GT_image[y:y+patch_height, x:x+patch_width]   # (ph, pw, 3)

                result = ray_march(
                    ray_origins    = ro,
                    ray_directions = rd,
                    sampler        = sampler,
                    tfn_lut        = tfn_lut,
                    cfg            = cfg,
                    tfn_file       = None,
                    scene_aabb     = scene_aabb,
                    light_dir_normalized=None,
                    nets = nets
                )
                
                # lpips expect the channel dim to be the first
                # and expect the value range to be -1~1
                loss = lpips(result.permute(2, 0, 1).unsqueeze(0) * 2 - 1, gt.permute(2, 0, 1).unsqueeze(0) * 2 - 1)
                optimizer.zero_grad()
                loss.backward()          # graph freed here
                optimizer.step()
                print(f"epoch: {epoch}, iteration: {iteration}, loss: {loss.item()}")
                tensorboard_writer.add_scalar("loss", loss.item(), iteration)
                iteration += 1
                
                if epoch == epochs - 1 or epoch % 50 == 0 or epoch < 10:
                    full_images[y:y+patch_height, x:x+patch_width] = result
                    
         
        if epoch % 50 == 0 or epoch < 10:
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
                    
                
    return full_images

# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cam = Camera(
        # TODO: see how can I convert camera parameters from world space to object space
        # for example, the object (mechhand) in world space is (0,0,0)~(640,220,229), and its object space is (0,0,0)~(1,1,1)
        # for mechhand
        # position = torch.tensor([0.0, 1.5,  1.5]),
        # look_at  = torch.tensor([0.5, 0.5,  0.5]),
        # up       = torch.tensor([0.0, 1.0,  0.0]),
        # for spider
        position = torch.tensor([-0.085, -0.31, -0.012]),
        look_at  = torch.tensor([0.45, 0.4,  0.5]),
        up       = torch.tensor([0.1, 0.342,  -0.93]),
        # position = torch.tensor([-1.0394, 2.5554, 2.5044]),
        # look_at  = torch.tensor([0.5-0.004525, 0.5-0.3158,  1.1093]),
        # up       = torch.tensor([0.3800, 0.8572,  -0.341]),
        fov_y    = 60.0,
        width    = 384,
        height   = 384,
    )

    aabb = torch.tensor([[0., 0., 0.], [1., 1., 1.]])

    cfg = MarchConfig(
        # for mechhand
        # t_near    = 0.1,
        # t_far     = 2.8,
        # n_samples = 1024,
        # for spider
        # NOTE: might be okay to clip volume a little bit as we just want the majority of the rendered volume for calculating rendering loss
        t_near    = 0.6,
        t_far     = 1.3,
        n_samples = 1024,
        patch_width=32,
        patch_height=32,
    )

    # raw_data_file_path = "/home/kctung/Projects/BlockFusion/datasets/MechHand_f_640x220x229.raw"
    # tfn_file_path = "/home/kctung/Projects/BlockFusion/datasets/scene_mechhand.json"
    raw_data_file_path = "/home/kctung/Projects/BlockFusion/datasets/zea_c_spider2_957x1195x1003_float32.raw"
    tfn_file_path = "/home/kctung/Projects/BlockFusion/datasets/scene_spider_transparent_pink.json"

    # sample light direction normalized
    # for mechhand 
    # light_dir_normalized = [0.25, 0.25]
    # for spider
    # light_dir_normalized = [0.76, 0.4]
    light_dir_normalized = [0.555, 0.827]

    with open(tfn_file_path, 'r') as f:
        tfn_json = json5.load(f)
    
    resolution = tfn_json["dataSource"][0]["dimensions"]
    resolution = [resolution["x"], resolution["y"], resolution["z"]]
    colorControls = tfn_json["view"]["volume"]["transferFunction"]["colorControls"]
    opacityControl = tfn_json["view"]["volume"]["transferFunction"]["opacityControl"]

    sampler = create_sampler("structuredRegular", "cuda", dims=resolution, dtype="float32", n_channels=1, filename=raw_data_file_path)

    with open("shadows_subset_training_SIREN.json", 'r') as f:
        config = json5.load(f)
    config = edict(config)
        
    loaded_model = torch.load("/home/kctung/Projects/HyperDiffusion/logs/spider_2000_embd720_head16_layer12/2026-02-25_06-02-52/sample_siren_7000_test.pt", map_location="cpu")
    
    n_instances = len(loaded_model['light_dir_cartesian'])
    
    GT_image = torch.load("/home/kctung/Projects/BlockFusion/logs/rendering_loss_dev_exp/GT_image.pt")["GT_image"]

    nets = [NeurCompNet(n_input_dims=3, 
                    n_output_dims=config.n_labels, bias=False, 
                    n_hidden_layers=config.n_layers, 
                    n_neurons=config.n_hid, is_residual=True).cuda() for _ in range(n_instances)]
    nets = torch.nn.ModuleList(nets)
    nets.load_state_dict(loaded_model['net_state_dict'])
    nets.train()
    net_0 = nets[0]

    save_dir = "/home/kctung/Projects/BlockFusion/logs/rendering_loss_dev_exp"
    tfn_lut = build_transfer_function(colorControls, opacityControl, lut_size=1024)
    result = patchify_render_training(cam, sampler, tfn_lut, scene_aabb=aabb, cfg=cfg, device=device, GT_image=GT_image, nets=net_0, save_dir=save_dir)
    print("Smoke test passed.")
    print(f"max memory allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"max memory reserved: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")
    
    
    data = result
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
    plt.title("Full render with shadow after finetuning with rendering loss")
    plt.savefig(os.path.join(save_dir,"render_after_finetune.png"))
    plt.close()
    
        # to generate GT data
    finetune_result_to_save = dict()
    # HACK: only one sample now
    finetune_result_to_save["finetuned_image"] = result
    finetune_result_to_save["light_dir_spherical_normalized"] = light_dir_normalized
    finetune_result_to_save["light_dir_cartesian"] = loaded_model['light_dir_cartesian']
    finetune_result_to_save['net_state_dict'] = nets.state_dict()
    # save image tensor as pt file
    torch.save(finetune_result_to_save, os.path.join(save_dir, "finetuned_image.pt"))