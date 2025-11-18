# imports
import json

import torch
import utils
from torch.utils.data import DataLoader

# filling number (1 or 2)
fi = 1

# get training raw data
tensor_list = []
tensor_list.append(
    torch.load(
        f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tunable_interaction_range_datasets3/filling_{fi}/train_6x6.pt",
        weights_only=True,
    )
)
tensor_list.append(
    torch.load(
        f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tunable_interaction_range_datasets3/filling_{fi}/train_8x8.pt",
        weights_only=True,
    )
)

# make combined dataset
training_split = utils.combine_datasets(tensor_list)
print(f"\nTraining Dataset Shape: {training_split.shape}\n")

# preprocessing
training_split = utils.normalize_dataset(training_split)

# make sure it's normalized properly
# mean and standard deviation should be close to 0 and 1, respectively
print("\nNew mean and standard deviations, respectively:")
print(utils.get_stats(training_split[:, 0, :, :]))
print(utils.get_stats(training_split[:, 1, :, :]))

# build training datasets
training_data = utils.HartreeFockDataset(training_split, type="full")
print(f"\nTraining dataset length: {training_data.__len__()}\n")

# Now time for the test dataset
# First, gather the data
eval_list = []
dims = [20, 30, 40]
for dim in dims:
    # first, get the dataset
    eval_dataset = torch.load(
        f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tunable_interaction_range_datasets3/filling_{fi}/test_{dim}x{dim}.pt",
        weights_only=True,
    )
    cutoff = min(len(eval_dataset), 500)
    eval_dataset = eval_dataset[:cutoff]
    eval_list.append(eval_dataset)
eval_split = utils.combine_datasets(eval_list)
print(f"Evaluation Dataset Shape: {eval_split.shape}\n")

# Now we need to normalize it
# load normalization constants from file
with open("normalization_constants.json", "r") as f:
    normalization_constants = json.load(f)

x_mean = normalization_constants["x_mean"]
x_std = normalization_constants["x_std"]
y_mean = normalization_constants["y_mean"]
y_std = normalization_constants["y_std"]

# normalize dataset using the same constants
eval_split[:, 0, :, :] = utils.normalize(eval_split[:, 0, :, :], x_mean, x_std)
eval_split[:, 1, :, :] = utils.normalize(eval_split[:, 1, :, :], y_mean, y_std)

# Finally, pack everything into the dataset class
test_data = utils.HartreeFockDataset(eval_split, type="full")
print(f"\nTest dataset length: {test_data.__len__()}\n")

# training and validation dataloaders
train_dataloader = DataLoader(
    training_data, batch_size=64, shuffle=True, pin_memory=True, num_workers=3
)
val_dataloader = DataLoader(
    test_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=3
)

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# instantiate model
model = utils.HF_SimpleModel(
    num_layers=3, input_dim=32, num_heads=2, dim_feedforward=256, dropout=0.0, device=device
)

# Loss function and optimizer
loss_fn = utils.masked_MSE_loss
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

train_losses = []
val_losses = []
epochs = 300

# train the model and record the loss for the training and validation datasets
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_losses.append(
        utils.train_loop(train_dataloader, model, loss_fn, optimizer, device)
    )
    val_losses.append(utils.test_loop(val_dataloader, model, loss_fn, device))
print("Done!\n")

print(f"Training Losses (1-{epochs}): {train_losses}")
print(f"Test Losses (1-{epochs}): {val_losses}\n")

utils.plot_losses(train_losses, val_losses)

# saving model
torch.save(
    model.state_dict(),
    "tunable_interaction_range3_6x6_8x8_model.pth",
)
