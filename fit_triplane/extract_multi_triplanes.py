import torch
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str)
parser.add_argument("--filename", type=str)
# parser.add_argument("--indices", type=int, nargs='*', default=[500, 501])
parser.add_argument("--permuted_indices_filepath", type=str)
args = parser.parse_args()

# 100 random indices
# indices=[40, 42, 44, 50, 116, 124, 127, 159, 184, 194, 215, 239, 255, 282, 311, 325,
#  332, 378, 396, 409, 414, 438, 527, 567, 624, 636, 679, 699, 721, 748, 807,
#  808, 819, 852, 889, 902, 913, 926, 961, 1001, 1021, 1023, 1034, 1036, 1059,
#  1074, 1084, 1087, 1100, 1117, 1179, 1185, 1214, 1217, 1218, 1238, 1244, 1297,
#  1301, 1325, 1375, 1390, 1410, 1449, 1458, 1476, 1477, 1507, 1508, 1518, 1522,
#  1558, 1582, 1588, 1590, 1640, 1676, 1677, 1724, 1732, 1735, 1752, 1763, 1767,
#  1769, 1800, 1805, 1816, 1826, 1841, 1853, 1879, 1909, 1935, 1961, 2000, 2008,
#  2068, 2071, 2091]
# indices = [500, 501]
# indices = [40, 740, 1440]
indices = [50, 116, 124, 889, 902, 913, 1458, 1476, 1477]
# indices = [1061]

old_model = torch.load(os.path.join(args.model_dir, args.filename))
new_model = dict()
if "permuted_indices" in old_model:
    permuted_indices = old_model['permuted_indices']
else:
    if args.permuted_indices_filepath is None:
        raise RuntimeError("Please provide permuted indices filepath.")
    else:
        permuted_indices = torch.load(args.permuted_indices_filepath)['permuted_indices']
    
inverse_indices = np.argsort(permuted_indices)
mapped_indices = [inverse_indices[i] for i in indices]

new_model['net_state_dict'] = old_model['net_state_dict']
new_model['light_dir_cartesian'] = [old_model['light_dir_cartesian'][i] for i in mapped_indices]

new_model['triplane_state_dict'] = dict()
for batch_idx, idx in enumerate(mapped_indices):
    new_model['triplane_state_dict'][f'{batch_idx}.triplane'] = old_model['triplane_state_dict'][f'{idx}.triplane'].clone()
    new_model['triplane_state_dict'][f'{batch_idx}.plane_axes'] = old_model['triplane_state_dict'][f'{idx}.plane_axes'].clone()

torch.save(new_model, os.path.join(args.model_dir, f"VAE_recon_tri_{len(mapped_indices)}_ins.pt"))