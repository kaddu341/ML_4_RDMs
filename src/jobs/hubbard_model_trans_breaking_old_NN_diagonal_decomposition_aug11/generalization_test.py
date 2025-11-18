import json

import matplotlib.pyplot as plt
import torch
import utils
from torch import nn
from torch.utils.data import DataLoader

# ### Evaluate generalization capabilities

# set up x and y lists
losses = []

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# get the filenames
dims = list(range(3, 19, 3))    # should go from 3 to 18 in increments of 3
filenames = [f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/BZindex_{d}x{d}.pt" for d in dims]

# load model (make sure matches what's in the folder)
model = utils.HF_SimpleModel(
    filenames,
    num_layers = 3,
    input_dim = 32,
    embed_dim = 64,
    output_dim = 32,
    num_heads = 4,
    num_freqs = 96,
    dim_feedforward = 128,
    input_dropout = 0.0,
    output_dropout = 0.0,
    dropout = 0.0,
    device = device
)

# Load the saved state_dict
# state_dict = torch.load(dir + "tMoTe2_2BPV_HF_model_num_freqs_256.pth")
state_dict = torch.load("/blue/yujiabin/awwab.azam/hartree-fock-code/src/jobs/tMoTe2_6x6_9x9_freq_test_lc_july5/96_test_index_44_keep/tMoTe2_2BPV_HF_model_num_freqs_96_final.pth")

# Create a new state_dict without the prefix
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith('_orig_mod.'):
        name = k[10:] # remove `_orig_mod.`
        new_state_dict[name] = v
    else:
        new_state_dict[k] = v

# Load the corrected state_dict
model.load_state_dict(new_state_dict)

# get number of trainable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

# load normalization constants from file
with open("normalization_constants.json", "r") as f:
    normalization_constants = json.load(f)

x_mean = normalization_constants["x_mean"]
x_std = normalization_constants["x_std"]
y_mean = normalization_constants["y_mean"]
y_std = normalization_constants["y_std"]

print("Normalization Constants:")
print(f"\nInputs Mean: {x_mean}, Inputs Standard Deviation: {x_std}")
print(f"Targets Mean: {y_mean}, Targets Standard Deviation: {y_std}")

print("Commencing generalization test...")
for dim in dims:
    # first, get the dataset
    filename = f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/tMoTe2_dataset_{dim}x{dim}_10K.h5"

    # load dataset
    eval_dataset = utils.HF_3x3_Dataset(filename, normalization_constants, indices=(9_000, 10_000))
    print(f"\nDataset size: {eval_dataset.__len__()}")

    # make a new DataLoader for testing purposes
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=100, shuffle=True, pin_memory=True, num_workers=1
    )

    # get loss for the new model on this dataset
    loss = utils.test_loop([eval_dataloader], model, torch.nn.MSELoss(reduction = "sum"), device)
    losses.append(loss)

print(losses)