import os
from autoencoder import *
from timevarying_data_helper import TimevaryingDataset, EncodingWeightDataset
import logging
import argparse
from datetime import datetime
# model configs are stored as python scripts, import the target config here
from autoencoder_config import model_a as cfg

def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE on time-varying data")
    # parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train")
    parser.add_argument("--expdir", type=str, default="./logs/test_hashencoding_train/20250508-024915", help="Checkpoint Directory to load the model from")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    vae_model = VAE(cfg.vae_config, cfg.encoder_dims, cfg.feature_size_encoder, cfg.decoder_dims, cfg.feature_size_decoder, cfg.fpn_encoders_layer_dim_idx,
                    cfg.fpn_decoders_layer_dim_idx, cfg.fpn_encoders_down_idx, cfg.fpn_encoders_up_idx, cfg.fpn_decoders_down_idx, cfg.fpn_decoders_up_idx, cfg.block_config)
    vae_model = torch.nn.DataParallel(vae_model)
    vae_model = vae_model.cuda()
    vae_model.load_state_dict(torch.load(os.path.join(args.expdir, "vae_model_epoch_39999.ckpt"))["model_state_dict"])
    
    # # prepare dataset
    pretrained_weights = torch.load("/home/kctung/Projects/instant-vnr-pytorch/logs/hyperinr/debug/run00028/checkpoint-last.ckpt")
    dataset = EncodingWeightDataset(
            pretrained_weights_info=pretrained_weights["model_state_dict"],
            level=1)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        # batch_size=args.batch_size,
        batch_size=cfg.batch_size,
        shuffle=False)
    
    
    test_data = next(iter(train_dataloader))
    # test_data = torch.randn(2, 8, 32, 32, 32).cuda()
    print("test data shape: {}".format(test_data.shape))
    out = vae_model(test_data)
    
    # loss = vae_model.module.loss_function(*out)
    # print("loss: {}".format(loss))
    print("z shape: {}".format(out[-1].shape))
    print("reconstruct shape: {}".format(out[0].shape))
    
    outputs = []
    
    for batch_idx, raw_data in enumerate(train_dataloader):
        output = vae_model(raw_data)
        # reconstructed results is the first element of the output (output[0])
        recon_loss = F.mse_loss(output[0], raw_data)
        # kl_loss = vae_model.module.loss_function(*output)
        # loss = recon_loss + kl_loss
        loss = recon_loss
        # print(f"Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, KL loss: {kl_loss.item():0,.6f}, Reconstruction PSNR: {(20 * torch.log10(raw_data.max() - raw_data.min() / torch.sqrt(recon_loss))):0,.4f}")
        print(f"Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, Reconstruction PSNR: {(20 * torch.log10(raw_data.max() - raw_data.min() / torch.sqrt(recon_loss))):0,.4f}")
        outputs.append(output[0])
    
    outputs = torch.cat(outputs, dim=0)
    replaced_weights = dataset.replace_level_n_weights(outputs)
    # unsqueeze a dim to comply with the input shape of reconstruction code in instant-vnr-pytorch
    replaced_weights = replaced_weights.unsqueeze(0)
    torch.save({"generated_weights_samples": replaced_weights}, f"{args.expdir}/generated_weights_samples.pt")
    
    
    # tensorboard_writer.close()
    # # samples = vae_model.sample(2)
    # # print("samples shape: {}".format(samples[0].shape))