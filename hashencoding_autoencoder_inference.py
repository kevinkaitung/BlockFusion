import os
from autoencoder import *
from timevarying_data_helper import TimevaryingDataset, EncodingWeightDataset
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
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
    vae_model.load_state_dict(torch.load(os.path.join(args.expdir, "vae_model_epoch_24999.ckpt"))["model_state_dict"])
    
    # # prepare dataset
    pretrained_weights = torch.load("/home/kctung/Projects/instant-vnr-pytorch/logs/hyperinr/debug/run00028/checkpoint-last.ckpt")
    dataset = EncodingWeightDataset(
            pretrained_weights_info=pretrained_weights["model_state_dict"],
            level=1)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)
    
    
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
        output = vae_model(raw_data)
        # reconstructed results is the first element of the output (output[0])
        recon_loss = F.mse_loss(output[0], raw_data)
        kl_loss = vae_model.module.loss_function(*output)
        loss = recon_loss + kl_loss
        # loss = recon_loss
        print(f"Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, KL loss: {kl_loss.item():0,.6f}, Reconstruction PSNR: {(20 * torch.log10(raw_data.max() - raw_data.min() / torch.sqrt(recon_loss))):0,.4f}")
        # print(f"Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, Reconstruction PSNR: {(20 * torch.log10(raw_data.max() - raw_data.min() / torch.sqrt(recon_loss))):0,.4f}")
        outputs.append(output[0])
    
    outputs = torch.cat(outputs, dim=0)
    replaced_weights = dataset.replace_level_n_weights(outputs)
    # unsqueeze a dim to comply with the input shape of reconstruction code in instant-vnr-pytorch
    replaced_weights = replaced_weights.unsqueeze(0)
    torch.save({"generated_weights_samples": replaced_weights}, f"{args.expdir}/generated_weights_samples.pt")
    
    
    # tensorboard_writer.close()
    # # samples = vae_model.sample(2)
    # # print("samples shape: {}".format(samples[0].shape))