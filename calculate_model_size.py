import torch
import torch.nn as nn
import argparse
import importlib
from autoencoder_2D_origin import VAE_no_KL

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Model Sizes")
    parser.add_argument("--model_configs", nargs='+', type=str, help="Model config file name")
    return parser.parse_args()

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def model_size_in_mb(model):
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_bytes = param_size_bytes + buffer_size_bytes
    return total_size_bytes / (1024 ** 2)  # Convert to MB

if __name__ == "__main__":
    args = parse_args()
    model_configs = args.model_configs
    total_params_all_models = []
    trainable_params_all_models = []
    size_mbs_all_models = []
    for idx, model_config in enumerate(model_configs):
        cfg = importlib.import_module(f"autoencoder_config.triplane.{model_config}")
        vae_model = torch.nn.DataParallel(VAE_no_KL(cfg.vae_config, cfg.encoder_dims, cfg.feature_size_encoder, cfg.decoder_dims,
                                                    cfg.feature_size_decoder, cfg.fpn_encoders_layer_dim_idx, cfg.fpn_decoders_layer_dim_idx,
                                                    cfg.fpn_encoders_down_idx, cfg.fpn_encoders_up_idx, cfg.fpn_decoders_down_idx,
                                                    cfg.fpn_decoders_up_idx, cfg.block_config)).cuda()
        
        total, trainable = count_parameters(vae_model)
        size_mb = model_size_in_mb(vae_model)
        
        total_params_all_models.append(total)
        trainable_params_all_models.append(trainable)
        size_mbs_all_models.append(size_mb)
        
    # after gathering the information of model sizes, print it out
    for idx, model_config in enumerate(model_configs):
        print("-------------------------------------------------------")
        print(f"Model config name:    {model_config}")
        print(f"Total parameters:     {total_params_all_models[idx]:,}")
        print(f"Trainable parameters: {trainable_params_all_models[idx]:,}")
        print(f"Model size:           {size_mbs_all_models[idx]:.2f} MB")