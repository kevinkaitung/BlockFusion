import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str)
parser.add_argument("--filename", type=str)
# parser.add_argument("--indices", type=int, nargs='*', default=[500, 501])
# parser.add_argument("--permuted_indices_filepath", type=str, default=None)
parser.add_argument("--new_filename", type=str)
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
indices=[idx for idx in range(64)]
# indices = [527, 472, 445, 399, 464, 156, 444, 266, 109, 342, 283, 67, 345, 550, 230, 506]
print(indices)

old_model = torch.load(os.path.join(args.model_dir, args.filename))
new_model = dict()

# used to find the original indices given the indices in the permuted array
# if args.permuted_indices_filepath:
#     original_indices = []
#     permuted_indices = torch.load(args.permuted_indices_filepath)['permuted_indices']
#     for i in indices:
#         print(permuted_indices[i], end=" ")
#         original_indices.append(permuted_indices[i])
#     new_model['indices_in_original_order'] = original_indices

example_network = old_model['net_state_dict']
if 'light_dir_cartesian' in old_model.keys():
    new_model['light_dir_cartesian'] = [old_model['light_dir_cartesian'][i] for i in indices]
elif 'timesteps' in old_model.keys():
    new_model['timesteps'] = [old_model['timesteps'][i] for i in indices]

if 'pre_sampled_coord_groups' in old_model.keys():
    new_model['pre_sampled_coord_groups'] = [old_model['pre_sampled_coord_groups'][i] for i in indices]
    new_model['pre_sampled_value_groups'] = [old_model['pre_sampled_value_groups'][i] for i in indices]
if 'pre_cal_GT_images' in old_model.keys():
    new_model['pre_cal_GT_images'] = [old_model['pre_cal_GT_images'][i] for i in indices]
    new_model['camera_configs'] = old_model['camera_configs']
    new_model['aabb_configs'] = old_model['aabb_configs']
    new_model['march_configs'] = old_model['march_configs']

# get the layer keys first
layer_keys = []
for k, v in example_network.items():
    if k.startswith('0.'):
        idx_str, layer_name = k.split(".", 1)
        layer_keys.append(layer_name)

new_model['net_state_dict'] = dict()
for batch_idx, idx in enumerate(indices):
    for layer_key in layer_keys:
        new_model['net_state_dict'][f'{batch_idx}.{layer_key}'] = old_model['net_state_dict'][f'{idx}.{layer_key}'].clone()

torch.save(new_model, os.path.join(args.model_dir, f"{args.new_filename}.pt"))