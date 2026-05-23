from calculate_cam_params import *

if __name__ == "__main__":
    
    fov = 60.0
    image_width = 256
    image_height = 256
    
    base_dir = "/home/kctung/Projects/BlockFusion/fit_triplane/spider_multi_view_small_overlap_exps"
    
    with open(os.path.join(base_dir, "camera_params.json"), 'r') as f:
        json_data = json5.load(f)
    
    for batch_idx, key in enumerate(json_data.keys()):
        print("Processing batch idx: ", batch_idx)
        this_group_dir = os.path.join(base_dir, key)
        
        poses_tensors = []
        jpeg_tensors_this_instance = []
        for batch_j, each_camera_param in enumerate(json_data[key]):
            camera_position = each_camera_param["camera_position"]
            look_at = each_camera_param["origin"]
            up = each_camera_param["up"]
            
            # HACK: to offset camera to right position
            camera_position = [tmp + 478.5 for tmp in camera_position]
            look_at = [tmp + 597.5 for tmp in look_at]
            
            image_path = os.path.join(this_group_dir, f"view_{batch_j}.jpg")
            
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

        torch.save(scenes, os.path.join(this_group_dir, "chunk000.torch"))