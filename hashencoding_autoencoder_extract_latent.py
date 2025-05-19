import os
from autoencoder import *
from timevarying_data_helper import TimevaryingDataset, EncodingWeightDataset, LatentWeightDataset
import logging
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE on time-varying data")
    parser.add_argument("--expname", type=str, default="VAE_training_on_raw_volumes", help="Experiment name")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--expdir", type=str, default="./logs/test_hashencoding_train/20250508-024915", help="Checkpoint Directory to load the model from")
    parser.add_argument("--encode_or_decode", type=str, default="encode", help="Extract latent if encode; map latent back to its original weights if decode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.encode_or_decode == "encode":
        is_encode = True
    else:
        is_encode = False
    
    # setup VAE model hyperparameters
    vae_config = {"kl_std": 0.25,
                  "kl_weight": 0.001,
                  # 3 planes (xy, yz, xz) * 32 channels (feature vectors) * 128x128
                  "plane_shape": [args.batch_size, 8, 32, 32, 32],
                  "z_shape": [4, 16, 16, 16],
                  "num_heads": 16,
                  "transform_depth": 1}

    # encoder_in_channels = 64
    #                   idx:0,   1,   2,   3,   4,  5,   6,  7,     8,     9
    encoder_dims =         [32, 64, 128, 256, 512, 1024, 512, 256, 128,  2 * vae_config["z_shape"][0]]
    # encoder_dims =         [32, 64, 128, 256, 512, 1024, 512, 256, 128,  vae_config["z_shape"][0]]
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
    

    vae_model = VAE(vae_config, encoder_dims, feature_size_encoder, decoder_dims, feature_size_decoder, fpn_encoders_layer_dim_idx,
                    fpn_decoders_layer_dim_idx, fpn_encoders_down_idx, fpn_encoders_up_idx, fpn_decoders_down_idx, fpn_decoders_up_idx, block_config)
    vae_model = torch.nn.DataParallel(vae_model)
    vae_model = vae_model.cuda()
    vae_model.load_state_dict(torch.load(os.path.join(args.expdir, "vae_model_epoch_9999.ckpt"))["model_state_dict"])
    
    # # prepare dataset
    pretrained_weights = torch.load("/home/kctung/Projects/instant-vnr-pytorch/logs/hyperinr/debug/run00028/checkpoint-last.ckpt")
    if not is_encode:
        # current latent_weights shape is torch.Size([16, 16384]), 16 is the number of samples generated by diffusion model
        # we only use one sample to compare with all time steps volumes to first see the generalizability of diffusion-generated weights
        latent_weights = torch.load("/home/kctung/Projects/HyperDiffusion/generated_weights_samples/dummy-0ewqmant/generated_weights_samples.pt")
    
    if is_encode:
        dataset = EncodingWeightDataset(
                pretrained_weights_info=pretrained_weights["model_state_dict"],
                level=1)
    else:
        original_hashencoding_dataset = EncodingWeightDataset(
                pretrained_weights_info=pretrained_weights["model_state_dict"],
                level=1)
        dataset = LatentWeightDataset(
            # currently only use the first sample
            # TODO: 
            latent_weights["generated_weights_samples"][:args.batch_size], vae_config["z_shape"])
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)
    
    if is_encode:
        test_data = next(iter(train_dataloader))
        # test_data = torch.randn(2, 8, 32, 32, 32).cuda()
        print("test data shape: {}".format(test_data.shape))
        out = vae_model(test_data)
        
        loss = vae_model.module.loss_function(*out)
        print("loss: {}".format(loss))
        print("z shape: {}".format(out[-1].shape))
        print("reconstruct shape: {}".format(out[0].shape))
    
    outputs = []
    
    for batch_idx, raw_data in enumerate(train_dataloader):
        if is_encode:
            output = vae_model.module.get_latent(raw_data)
        else:
            # TODO: not sure why using torch.nn.DataParallel for a single data point would cause error
            # so currently just use at least 2 for args.batch_size
            output = vae_model.module.decode(raw_data)
        outputs.append(output)
    
    outputs = torch.cat(outputs, dim=0)
    if is_encode:
        torch.save({"weights_latent_space": outputs, "z_shape": vae_config["z_shape"]}, f"{args.expdir}/weights_latent_space.pt")
    else:
        # temporarily copy output timesteps times since we only use one diffusion-generated sample
        # TODO:
        outputs = outputs[:1].repeat(len(original_hashencoding_dataset), 1, 1, 1, 1)
        # import pdb; pdb.set_trace()
        replaced_weights = original_hashencoding_dataset.replace_level_n_weights(outputs)
        replaced_weights = replaced_weights.unsqueeze(0)
        # import pdb; pdb.set_trace()
        torch.save({"generated_weights_samples": replaced_weights}, f"{args.expdir}/generated_weights_samples_from_latent.pt")
    
    # tensorboard_writer.close()
    # # samples = vae_model.sample(2)
    # # print("samples shape: {}".format(samples[0].shape))