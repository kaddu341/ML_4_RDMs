import json

import matplotlib.pyplot as plt
import torch
import utils
from torch import nn
from torch.utils.data import DataLoader

# filling number (1 or 2)
fi = 1

# ### Evaluate generalization capabilities

# set up x and y lists
losses = []
dims = list(range(6, 51, 2))

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load model
eval_model = utils.HF_SimpleModel(
    num_layers=3, input_dim=32, num_heads=2, dim_feedforward=256, dropout=0.0, device=device
)
eval_model.load_state_dict(
    torch.load(
        "tunable_interaction_range3_6x6_8x8_model.pth",
        weights_only=True,
    )
)

# load normalization constants from file
with open("normalization_constants.json", "r") as f:
    normalization_constants = json.load(f)

x_mean = normalization_constants["x_mean"]
x_std = normalization_constants["x_std"]
y_mean = normalization_constants["y_mean"]
y_std = normalization_constants["y_std"]

print(f"\nInputs Mean: {x_mean}, Inputs Standard Deviation: {x_std}")
print(f"Targets Mean: {y_mean}, Targets Standard Deviation: {y_std}")

print("Commencing generalization test...")
for dim in dims:
    # first, get the dataset
    if (dim == 6) or (dim == 8):
        filename = f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tunable_interaction_range_datasets3/filling_{fi}/train_{dim}x{dim}.pt"
    else:
        filename = f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tunable_interaction_range_datasets3/filling_{fi}/test_{dim}x{dim}.pt"
    
    eval_dataset = torch.load(filename, weights_only=True)
    
    cutoff = min(len(eval_dataset), 1000)
    eval_dataset = eval_dataset[:cutoff]
    print(f"\nDataset shape: {eval_dataset.shape}")

    # normalize dataset using the same constants
    eval_dataset[:, 0, :, :] = utils.normalize(eval_dataset[:, 0, :, :], x_mean, x_std)
    eval_dataset[:, 1, :, :] = utils.normalize(eval_dataset[:, 1, :, :], y_mean, y_std)

    # make a new dataset and DataLoader for testing purposes
    eval_dataset = utils.HartreeFockDataset(eval_dataset, type="full")
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=3
    )

    # get loss for the new model on this dataset
    # we can use normal MSE loss because there's no padding this time!
    loss = utils.test_loop(
        eval_dataloader, eval_model, loss_fn=nn.MSELoss(), device=device
    )
    losses.append(loss)

# # plot losses as a function of L
# plt.figure(figsize=(8, 5))
# plt.plot(dims, losses, marker="o")  # Plot the list with markers
# plt.title("Evaluation Losses")
# plt.xlabel("L")
# plt.ylabel("Value")
# plt.grid(True)

# # Save the plot to a file
# plt.savefig("generalization_test.png")  # Saves as PNG by default

# # Show the plot
# plt.show()

print(losses)
