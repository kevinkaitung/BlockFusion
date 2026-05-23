from render_GT_imgs_and_save_cam_params import *

def process_1_instance_camera_params_and_image(fov, image_width, image_height, camera_position, look_at, up, image_path):
    
    K = compute_intrinsics(fov_y_deg=fov, image_width=image_width, image_height=image_height)
    c2w = compute_extrinsics(camera_position, look_at, up)
    
    # HACK: scale down the camera position 
    # because FFGS framework might expect the scene sit within the range roughly [-1, 1] or [0, 1]
    # c2w[:3, 3] = c2w[:3, 3] / 500.0
    c2w[:3, 3] = c2w[:3, 3] / 750.0
    # c2w[:3, 3] = c2w[:3, 3] / 1000.0
    
    poses = build_poses_tensor(
        c2w,
        fx=K[0,0], fy=K[1,1],
        cx=K[0,2], cy=K[1,2],
    )

    # 1. Read image
    pil_img = Image.open(image_path).convert("RGB")
    # 2. Downsample to 256x256
    pil_img = pil_img.resize((256, 256), Image.LANCZOS)
    # 2. Convert to tensor in [0, 1]
    img_np = np.array(pil_img)                     # (H, W, 3), uint8
    img_tensor = torch.from_numpy(img_np).float() / 255.0
    # --------------------------------------------------
    # Now img_tensor mimics your GT_image tensor
    # --------------------------------------------------
    # 3. Convert back to uint8 image
    result_uint8 = (
        img_tensor.clamp(0, 1) * 255
    ).byte().cpu().numpy()
    # 4. Encode to JPEG bytes
    pil_img = Image.fromarray(result_uint8, mode="RGB")
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=95)
    jpeg_bytes = buffer.getvalue()
    # 5. Convert JPEG bytes -> uint8 tensor
    jpeg_tensor = torch.frombuffer(
        jpeg_bytes,
        dtype=torch.uint8
    )
    
    return poses, jpeg_tensor

if __name__ == "__main__":
    json_files = ["/home/kctung/Projects/BlockFusion/datasets/spider_multi_views/scene_spider_view_0.json",
                  "/home/kctung/Projects/BlockFusion/datasets/spider_multi_views/scene_spider_view_1.json",
                  "/home/kctung/Projects/BlockFusion/datasets/spider_multi_views/scene_spider_view_2.json",
                  "/home/kctung/Projects/BlockFusion/datasets/spider_multi_views/scene_spider_view_3.json",
                  "/home/kctung/Projects/BlockFusion/datasets/spider_multi_views/scene_spider_view_4.json",
                  "/home/kctung/Projects/BlockFusion/datasets/spider_multi_views/scene_spider_view_5.json",]
    
    image_paths = ["/home/kctung/Projects/instant-vnr-pytorch-new/instant-vnr-pytorch/screenshots/view_0.jpg",
                       "/home/kctung/Projects/instant-vnr-pytorch-new/instant-vnr-pytorch/screenshots/view_1.jpg",
                       "/home/kctung/Projects/instant-vnr-pytorch-new/instant-vnr-pytorch/screenshots/view_2.jpg",
                       "/home/kctung/Projects/instant-vnr-pytorch-new/instant-vnr-pytorch/screenshots/view_3.jpg",
                       "/home/kctung/Projects/instant-vnr-pytorch-new/instant-vnr-pytorch/screenshots/view_4.jpg",
                       "/home/kctung/Projects/instant-vnr-pytorch-new/instant-vnr-pytorch/screenshots/view_5.jpg",]
    
    fov = 60.0
    image_width = 256
    image_height = 256
    
    poses_tensors = []
    jpeg_tensors_this_instance = []
    for json_file, image_path in zip(json_files, image_paths):
        with open(json_file, 'r') as f:
            json_data = json5.load(f)
        
        camera_position = [json_data["view"]["camera"]["eye"]["x"], json_data["view"]["camera"]["eye"]["y"], json_data["view"]["camera"]["eye"]["z"]]
        look_at = [json_data["view"]["camera"]["center"]["x"], json_data["view"]["camera"]["center"]["y"], json_data["view"]["camera"]["center"]["z"]]
        up = [json_data["view"]["camera"]["up"]["x"], json_data["view"]["camera"]["up"]["y"], json_data["view"]["camera"]["up"]["z"]]
        
        poses, jpeg_tensor = process_1_instance_camera_params_and_image(fov, image_width, image_height, camera_position, look_at, up, image_path)
        
        poses_tensors.append(torch.tensor(poses))
        jpeg_tensors_this_instance.append(jpeg_tensor)

    poses_tensors = torch.stack(poses_tensors)
    
    scenes = []
    scene_id = 0
    scenes.append({
        "key": str(scene_id),
        "cameras": poses_tensors,  # shape [N, 18]: fx,fy,cx,cy, ..., 3x4 w2c
        "images": jpeg_tensors_this_instance  # list of raw JPEG bytes as uint8 tensors
    })
    scene_id += 1

    save_dir = "."
    torch.save(scenes, os.path.join(save_dir, "chunk000.torch"))