# Note: model tested before 5/20, named as bigger_net
# no fpn, no kl, no non-downsample/upsample CNN layers
batch_size = 90

# setup VAE model hyperparameters
vae_config = {
            "kl_std": 0.25,
            "kl_weight": 0.001,
            # 3 planes (xy, yz, xz) * 32 channels (feature vectors) * 128x128
            "plane_shape": [batch_size, 8, 32, 32, 32],
            "z_shape": [4, 16, 16, 16],
            "num_heads": 16,
            "transform_depth": 1
            }

# encoder_in_channels = 64
#                   idx:0,   1,   2,   3,   4,  5,   6,  7,     8,     9
# encoder_dims =         [32, 64, 128, 256, 512, 1024, 512, 256, 128,  2 * vae_config["z_shape"][0]]
encoder_dims =         [32, 64, 128, 256, 512, 1024, 512, 256, 128,  vae_config["z_shape"][0]]
feature_size_encoder = [32, 16,  8,   4,   2,   1,    2,   4,   8,  16]

# decoder_in_channels = 128
decoder_dims =         [32,  128, 256, 512, 1024, 512, 256, 128, 64, vae_config["plane_shape"][1]]
feature_size_decoder = [16,   8,   4,   2,   1,    2,   4,   8,  16, vae_config["plane_shape"][2]]

# these indices index for encoder_dims/decoder_dims
fpn_encoders_layer_dim_idx = []
fpn_decoders_layer_dim_idx = []

# these indices index for the group of blocks (i.e., encoders_down, ...) in block_config
fpn_encoders_down_idx = []
fpn_encoders_up_idx = []
fpn_decoders_down_idx = []
fpn_decoders_up_idx = []

block_config = {
    "encoders_down": [
                        {"in_channels":encoder_dims[0], "inter_channels":encoder_dims[1], "stride":2,
                        "out_channels":encoder_dims[1], "feature_size":feature_size_encoder[1], 
                        "use_transformer":False, "use_resblock":True, "is_decoder_output": False},
                        {"in_channels":encoder_dims[1], "inter_channels":encoder_dims[2], "stride":2,
                        "out_channels":encoder_dims[2], "feature_size":feature_size_encoder[2], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                        {"in_channels":encoder_dims[2], "inter_channels":encoder_dims[3], "stride":2,
                        "out_channels":encoder_dims[3], "feature_size":feature_size_encoder[3], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                        {"in_channels":encoder_dims[3], "inter_channels":encoder_dims[4], "stride":2,
                        "out_channels":encoder_dims[4], "feature_size":feature_size_encoder[4], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                        {"in_channels":encoder_dims[4], "inter_channels":encoder_dims[5], "stride":2,
                        "out_channels":encoder_dims[5], "feature_size":feature_size_encoder[5], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                    ],
    "encoders_up": [
                    # twice wider of input channels for FPN layer input
                    {"in_channels":encoder_dims[5], "inter_channels":encoder_dims[6], "stride":2,
                        "out_channels":encoder_dims[6], "feature_size":feature_size_encoder[6], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                    {"in_channels":encoder_dims[6], "inter_channels":encoder_dims[7], "stride":2,
                        "out_channels":encoder_dims[7], "feature_size":feature_size_encoder[7], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                    {"in_channels":encoder_dims[7], "inter_channels":encoder_dims[8], "stride":2,
                        "out_channels":encoder_dims[8], "feature_size":feature_size_encoder[8], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                        {"in_channels":encoder_dims[8], "inter_channels":encoder_dims[9], "stride":2,
                        "out_channels":encoder_dims[9], "feature_size":feature_size_encoder[9], 
                        "use_transformer":False, "use_resblock":True, "is_decoder_output": False}
                    ],
    "decoders_down": [
                    {"in_channels":decoder_dims[0], "inter_channels":decoder_dims[1], "stride":2,
                        "out_channels":decoder_dims[1], "feature_size":feature_size_decoder[1], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                        {"in_channels":decoder_dims[1], "inter_channels":decoder_dims[2], "stride":2,
                        "out_channels":decoder_dims[2], "feature_size":feature_size_decoder[2], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                    {"in_channels":decoder_dims[2], "inter_channels":decoder_dims[3], "stride":2,
                        "out_channels":decoder_dims[3], "feature_size":feature_size_decoder[3], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                        {"in_channels":decoder_dims[3], "inter_channels":decoder_dims[4], "stride":2,
                        "out_channels":decoder_dims[4], "feature_size":feature_size_decoder[4], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                    ],
    "decoders_up": [
                    {"in_channels":decoder_dims[4], "inter_channels":decoder_dims[5], "stride":2,
                        "out_channels":decoder_dims[5], "feature_size":feature_size_decoder[5], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                    # twice wider of input channels for FPN layer input
                        {"in_channels":decoder_dims[5], "inter_channels":decoder_dims[6], "stride":2,
                        "out_channels":decoder_dims[6], "feature_size":feature_size_decoder[6], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                        {"in_channels":decoder_dims[6], "inter_channels":decoder_dims[7], "stride":2,
                        "out_channels":decoder_dims[7], "feature_size":feature_size_decoder[7], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                        {"in_channels":decoder_dims[7], "inter_channels":decoder_dims[8], "stride":2,
                        "out_channels":decoder_dims[8], "feature_size":feature_size_decoder[8], 
                        "use_transformer":True, "use_resblock":False, "is_decoder_output": False},
                        {"in_channels":decoder_dims[8], "inter_channels":decoder_dims[8], "stride":2,
                        "out_channels":decoder_dims[9], "feature_size":feature_size_decoder[9], 
                        "use_transformer":False, "use_resblock":True, "is_decoder_output": True},
                    ],
}