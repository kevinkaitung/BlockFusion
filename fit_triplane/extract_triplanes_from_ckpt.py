import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--model_file', type=str, default=None)
args = parser.parse_args()

loaded_model = torch.load(os.path.join(args.model_dir, args.model_file))
torch.save({
                'net_state_dict': loaded_model['net_state_dict'],
                'triplane_state_dict': loaded_model['triplane_state_dict'],
            }, os.path.join(args.model_dir, f"pure_triplane_model.pt"))