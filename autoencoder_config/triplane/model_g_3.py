# Note: model created at 7/2,
# improved from model_g_2 to add more residual layers in a resblock

# setup VAE model hyperparameters
vae_config = {
            "kl_std": 0.25,
            "kl_weight": 0.001,
            # 3 planes (xy, yz, xz) * 32 channels (feature vectors) * 128x128
            "plane_shape": [3, 32, 128, 128],
            "z_shape": [4, 32, 32],
            "num_heads": 16,
            "transform_depth": 1
            }

# encoder_in_channels = 64
#                   idx:0,   1,   2,   3,   4,  5,   6,  7,     8,     9
# encoder_dims =         [32, 64, 128, 256, 512, 1024, 512, 256, 128,  2 * vae_config["z_shape"][0]]
encoder_dims =         [128, 512, 512, 1024, 1024, 1024, 1024, 1024,  2 * vae_config["z_shape"][0]]
feature_size_encoder = [128,  64,  32,   16,    8,    4,    8,   16,  vae_config["z_shape"][1]]
# Note: None is just the placeholder to keep index align with other arrays
encoder_use_transformers = [None, False, True, True, True, True, True, True, False]
encoder_use_resblocks = [None, True, True, True, False, False, False, True, True]
encoder_num_resblocks = 2

encoders_down_end_idx = 5
encoders_up_end_idx = 8

# decoder_in_channels = 128
decoder_dims =         [512,  1024, 1024, 1024, 1024, 1024, 512, 512, vae_config["plane_shape"][1]]
feature_size_decoder = [ 32,   16,    8,    4,    8,   16,  32,  64, vae_config["plane_shape"][2]]
decoder_use_transformers = [None, True, True, True, True, True, True, True, False]
decoder_use_resblocks = [None, True, True, False, False, False, True, True, True]
decoder_num_resblocks = 2

decoders_down_end_idx = 3
decoders_up_end_idx = 8

# these indices index for encoder_dims/decoder_dims
fpn_encoders_layer_dim_idx = [1, 2, 3, 4]
fpn_decoders_layer_dim_idx = [0, 1, 2, 3]

# these indices index for the group of blocks (i.e., encoders_down, ...) in block_config
fpn_encoders_down_idx = [0, 1, 2, 3]
fpn_encoders_up_idx = [1, 2]
fpn_decoders_down_idx = [-1, 0, 1, 2]
fpn_decoders_up_idx = [1, 2, 3]

block_config = {}

# Configure encoders_down
block_config["encoders_down"] = []
for i in range(encoders_down_end_idx):
    block_config["encoders_down"].append({
        "in_channels": encoder_dims[i],
        "inter_channels": encoder_dims[i+1],
        "stride": 2,
        "out_channels": encoder_dims[i+1],
        "feature_size": feature_size_encoder[i+1],
        "use_transformer": encoder_use_transformers[i + 1],
        "use_resblock": encoder_use_resblocks[i + 1],
        "is_decoder_output": False,
        "num_res_blocks": encoder_num_resblocks
    })

# Configure encoders_up
block_config["encoders_up"] = []
for j, i in enumerate(range(encoders_down_end_idx, encoders_up_end_idx)):
    block_config["encoders_up"].append({
        "in_channels": encoder_dims[i] * 2 if j in fpn_encoders_up_idx else encoder_dims[i],
        "inter_channels": encoder_dims[i+1],
        "stride": 2,
        "out_channels": encoder_dims[i+1],
        "feature_size": feature_size_encoder[i+1],
        "use_transformer": encoder_use_transformers[i + 1],
        "use_resblock": encoder_use_resblocks[i + 1],
        "is_decoder_output": False,
        "num_res_blocks": encoder_num_resblocks
    })

# Configure decoders_down
block_config["decoders_down"] = []
for i in range(decoders_down_end_idx):
    block_config["decoders_down"].append({
        "in_channels": decoder_dims[i],
        "inter_channels": decoder_dims[i+1],
        "stride": 2,
        "out_channels": decoder_dims[i+1],
        "feature_size": feature_size_decoder[i+1],
        "use_transformer": decoder_use_transformers[i + 1],
        "use_resblock": decoder_use_resblocks[i + 1],
        "is_decoder_output": False,
        "num_res_blocks": decoder_num_resblocks
    })

# Configure decoders_up
block_config["decoders_up"] = []
for j, i in enumerate(range(decoders_down_end_idx, decoders_up_end_idx)):
    block_config["decoders_up"].append({
        "in_channels": decoder_dims[i] * 2 if j in fpn_decoders_up_idx else decoder_dims[i],
        "inter_channels": decoder_dims[i] if i == (decoders_up_end_idx - 1) else decoder_dims[i+1],
        "stride": 2,
        "out_channels": decoder_dims[i+1],
        "feature_size": feature_size_decoder[i+1],
        "use_transformer": decoder_use_transformers[i + 1],
        "use_resblock": decoder_use_resblocks[i + 1],
        "is_decoder_output": True if i == (decoders_up_end_idx - 1) else False,
        "num_res_blocks": decoder_num_resblocks
    })

# block_config = {
#     "encoders_down": [
#                         {"in_channels":encoder_dims[0], "inter_channels":encoder_dims[1], "stride":2,
#                         "out_channels":encoder_dims[1], "feature_size":feature_size_encoder[1], 
#                         "use_transformer":False, "use_resblock":True, "is_decoder_output": False},
#                         {"in_channels":encoder_dims[1], "inter_channels":encoder_dims[2], "stride":2,
#                         "out_channels":encoder_dims[2], "feature_size":feature_size_encoder[2], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                         {"in_channels":encoder_dims[2], "inter_channels":encoder_dims[3], "stride":2,
#                         "out_channels":encoder_dims[3], "feature_size":feature_size_encoder[3], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                         {"in_channels":encoder_dims[3], "inter_channels":encoder_dims[4], "stride":2,
#                         "out_channels":encoder_dims[4], "feature_size":feature_size_encoder[4], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                         {"in_channels":encoder_dims[4], "inter_channels":encoder_dims[5], "stride":2,
#                         "out_channels":encoder_dims[5], "feature_size":feature_size_encoder[5], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                     ],
#     "encoders_up": [
#                     # twice wider of input channels for FPN layer input
#                     {"in_channels":encoder_dims[5], "inter_channels":encoder_dims[6], "stride":2,
#                         "out_channels":encoder_dims[6], "feature_size":feature_size_encoder[6], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                     {"in_channels":encoder_dims[6], "inter_channels":encoder_dims[7], "stride":2,
#                         "out_channels":encoder_dims[7], "feature_size":feature_size_encoder[7], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                     {"in_channels":encoder_dims[7], "inter_channels":encoder_dims[8], "stride":2,
#                         "out_channels":encoder_dims[8], "feature_size":feature_size_encoder[8], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                         {"in_channels":encoder_dims[8], "inter_channels":encoder_dims[9], "stride":2,
#                         "out_channels":encoder_dims[9], "feature_size":feature_size_encoder[9], 
#                         "use_transformer":False, "use_resblock":True, "is_decoder_output": False}
#                     ],
#     "decoders_down": [
#                     {"in_channels":decoder_dims[0], "inter_channels":decoder_dims[1], "stride":2,
#                         "out_channels":decoder_dims[1], "feature_size":feature_size_decoder[1], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                         {"in_channels":decoder_dims[1], "inter_channels":decoder_dims[2], "stride":2,
#                         "out_channels":decoder_dims[2], "feature_size":feature_size_decoder[2], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                     {"in_channels":decoder_dims[2], "inter_channels":decoder_dims[3], "stride":2,
#                         "out_channels":decoder_dims[3], "feature_size":feature_size_decoder[3], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                         {"in_channels":decoder_dims[3], "inter_channels":decoder_dims[4], "stride":2,
#                         "out_channels":decoder_dims[4], "feature_size":feature_size_decoder[4], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                     ],
#     "decoders_up": [
#                     {"in_channels":decoder_dims[4], "inter_channels":decoder_dims[5], "stride":2,
#                         "out_channels":decoder_dims[5], "feature_size":feature_size_decoder[5], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                     # twice wider of input channels for FPN layer input
#                         {"in_channels":decoder_dims[5], "inter_channels":decoder_dims[6], "stride":2,
#                         "out_channels":decoder_dims[6], "feature_size":feature_size_decoder[6], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                         {"in_channels":decoder_dims[6], "inter_channels":decoder_dims[7], "stride":2,
#                         "out_channels":decoder_dims[7], "feature_size":feature_size_decoder[7], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                         {"in_channels":decoder_dims[7], "inter_channels":decoder_dims[8], "stride":2,
#                         "out_channels":decoder_dims[8], "feature_size":feature_size_decoder[8], 
#                         "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
#                         {"in_channels":decoder_dims[8], "inter_channels":decoder_dims[8], "stride":2,
#                         "out_channels":decoder_dims[9], "feature_size":feature_size_decoder[9], 
#                         "use_transformer":False, "use_resblock":True, "is_decoder_output": True},
#                     ],
# }