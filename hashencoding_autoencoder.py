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
    parser.add_argument("--ckpt_freq", type=int, default=1000, help="Checkpoint frequency")
    parser.add_argument("--init_lr", type=float, default=1e-3, help="Initial learning rate for training")
    parser.add_argument("--lr_decay", type=int, default=3000, help="Learning rate decay frequency")
    parser.add_argument("--lr_gamma", type=float, default=0.2, help="Learning rate decay factor")
    return parser.parse_args()

# redefinition of traning pipeline for multiple input volumes
def train_vae(vae_model, train_dataloader, optimizer, epochs=100, tensorboard_writer=None, console_logger=None, run_dir=None, ckpt_freq=100):

    vae_model.train()
    
    # Add gradient scaling for mixed precision training
    # But loss would become NaN, so disable it for now
    scalar = torch.amp.GradScaler("cuda", enabled=False)
    
    for epoch in range(epochs):
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        running_loss = 0.0
        total_elems = 0
        
        min = float('inf')
        max = float('-inf')
        
        # mini-batch or SGD (with small batch as one sample) training
        # since we do optimization after each batch
        for batch_idx, raw_data in enumerate(train_dataloader):
            
            if raw_data.max() > max:
                max = raw_data.max()
            if raw_data.min() < min:
                min = raw_data.min()
            
            # Add gradient scaling for mixed precision training
            # But loss would become NaN, so disable it for now
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                output = vae_model(raw_data)
                # reconstructed results is the first element of the output (output[0])
                recon_loss = F.mse_loss(output[0], raw_data)
                kl_loss = vae_model.module.loss_function(*output)
                loss = recon_loss + kl_loss
                # loss = recon_loss
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            running_recon_loss += recon_loss.item() * raw_data.shape[0]
            running_kl_loss += kl_loss.item() * raw_data.shape[0]
            running_loss += running_recon_loss + running_kl_loss
            total_elems += raw_data.shape[0]
            
            # TODO: need to sperate PSNR evaluation from each volume (cause currently has four volumes in one batch)
            console_logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, KL loss: {kl_loss.item():0,.6f}, Reconstruction PSNR: {(20 * torch.log10(raw_data.max() - raw_data.min() / torch.sqrt(recon_loss))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
            # console_logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Total loss: {loss.item():0,.6f}, Recon loss: {recon_loss.item():0,.6f}, Reconstruction PSNR: {(20 * torch.log10(raw_data.max() - raw_data.min() / torch.sqrt(recon_loss))):0,.4f}, LR: {scheduler.get_last_lr()[0]}")
        
        val_range = max - min
        
        # calculate the average loss for the epoch
        assert total_elems == len(train_dataloader.dataset), "total_elems should be equal to the dataset size"
        last_recon_loss = running_recon_loss / total_elems
        last_kl_loss = running_kl_loss / total_elems
        last_loss = running_loss / total_elems
        last_PSNR = 20 * torch.log10(val_range / torch.sqrt(torch.tensor(last_recon_loss)))
        tensorboard_writer.add_scalar("Loss/Train_Recon", last_recon_loss, epoch)
        tensorboard_writer.add_scalar("Loss/Train_KL", last_kl_loss, epoch)
        tensorboard_writer.add_scalar("Loss/Train", last_loss, epoch)
        tensorboard_writer.add_scalar("Loss/Train_PSNR", last_PSNR, epoch)
        tensorboard_writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)
        
        # adjust learning rate
        scheduler.step()
        
        # save the model at checkpoint
        if (epoch % ckpt_freq == (ckpt_freq - 1)) or (epoch == epochs - 1):
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'loss': last_loss,
                'recon_loss': last_recon_loss,
                'kl_loss': last_kl_loss,
                'PSNR': last_PSNR,
            }, os.path.join(run_dir, f"vae_model_epoch_{epoch}.ckpt"))

            # save one of the reconstructed volume data     
            with open(os.path.join(run_dir, f"reconstructed_volume_epoch_{epoch}.data"), "wb") as f:
                # write the reconstructed volume data to file
                # only try to store the last batch's first volume
                output[0][0].clamp(raw_data.min(), raw_data.max()).detach().cpu().numpy().astype(np.float32).tofile(f)
        
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
    
    # # prepare dataset
    pretrained_weights = torch.load("/home/kctung/Projects/instant-vnr-pytorch/logs/hyperinr/debug/run00028/checkpoint-last.ckpt")
    train_dataloader = torch.utils.data.DataLoader(
            EncodingWeightDataset(
            pretrained_weights_info=pretrained_weights["model_state_dict"],
            level=1
        ),
        batch_size=args.batch_size,
        shuffle=True)
    
    # # create directory for saving logs
    base_dir = "./logs"
    os.makedirs(base_dir, exist_ok=True)
    expname_dir = os.path.join(base_dir, args.expname)
    os.makedirs(expname_dir, exist_ok=True)
    run_dir = os.path.join(expname_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    
    # # create tensorboard logger
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_writer = SummaryWriter(log_dir=run_dir)
    
    # # prepare python logger
    logging.basicConfig(filename=os.path.join(run_dir, "console_log.log"),
                    format='%(asctime)s %(message)s',
                    filemode='w')
    console_logger = logging.getLogger()
    console_logger.setLevel(logging.DEBUG)
    
    test_data = next(iter(train_dataloader))
    # test_data = torch.randn(2, 8, 32, 32, 32).cuda()
    print("test data shape: {}".format(test_data.shape))
    out = vae_model(test_data)
    # currently not working, because in order to draw the computational graph,
    # the model forward function should not has data-dependent control flow
    # tensorboard_writer.add_graph(vae_model.module, test_data.detach())
    loss = vae_model.module.loss_function(*out)
    print("loss: {}".format(loss))
    print("z shape: {}".format(out[-1].shape))
    print("reconstruct shape: {}".format(out[0].shape))
    
    # # save model architecture into tensorboard
    model_arch_str = str(vae_model)
    tensorboard_writer.add_text("Model/Architecture", f"```\n{model_arch_str}\n```", global_step=0)
    # console_logger.debug(f"Model architecture:\n{model_arch_str}")
    console_logger.debug(f"Batch Size: {args.batch_size}, Epochs: {args.epochs}, Checkpoint Frequency: {args.ckpt_freq}")
    console_logger.debug(f"Initial Learning rate: {args.init_lr}, Learning rate decay frequency: {args.lr_decay}, Learning rate decay factor: {args.lr_gamma}")
    
    # # test training autoencoder
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=args.init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.lr_gamma)
    train_vae(vae_model, train_dataloader, optimizer,
              args.epochs, tensorboard_writer, console_logger, run_dir, ckpt_freq=args.ckpt_freq)
    
    # tensorboard_writer.close()
    # # samples = vae_model.sample(2)
    # # print("samples shape: {}".format(samples[0].shape))