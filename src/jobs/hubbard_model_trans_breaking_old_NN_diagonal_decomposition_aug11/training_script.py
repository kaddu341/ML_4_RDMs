# imports
import json

import torch
import utils
from torch.utils.data import DataLoader


TRAIN_DATASET1 = "/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/tMoTe2_dataset_6x6_10K.h5"
TRAIN_DATASET2 = "/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/tMoTe2_dataset_9x9_10K.h5"
VAL_DATASET = "/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/tMoTe2_dataset_12x12_10K.h5"

# set batch size
BATCH_SIZE = 100

# for lr scheduler
MAX_LR = 1e-3
MAX_NUM_EPOCHS = 500
NUM_WARMUP_EPOCHS = 20


filenames_dict = {
    TRAIN_DATASET1: 10_000,
    TRAIN_DATASET2: 10_000,
}

# get normalization constants
x_mean, x_std, y_mean, y_std = utils.get_normalization_constants_3x3(filenames_dict)

print("Normalization Constants:")
print(f"\nInputs Mean: {x_mean}, Inputs Standard Deviation: {x_std}")
print(f"Targets Mean: {y_mean}, Targets Standard Deviation: {y_std}")

# get norm constants dict
with open("normalization_constants.json", "r") as f: 
    normalization_constants = json.load(f)

# make training and validation datasets
training_data1 = utils.HF_3x3_Dataset(TRAIN_DATASET1, normalization_constants)
print(f"Training dataset length: {training_data1.__len__()}\n")

training_data2 = utils.HF_3x3_Dataset(TRAIN_DATASET2, normalization_constants)
print(f"Training dataset length: {training_data2.__len__()}\n")

validation_data = utils.HF_3x3_Dataset(VAL_DATASET, normalization_constants, indices=(0, 5_000))
print(f"Test dataset length: {validation_data.__len__()}\n")

# training and test dataloaders
train_dataloader1 = DataLoader(training_data1, BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=1)
train_dataloader2 = DataLoader(training_data2, BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
val_dataloader = DataLoader(validation_data, BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

dims = list(range(3, 19, 3))    # should go from 3 to 18 in increments of 3
filenames = [f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/BZindex_{d}x{d}.pt" for d in dims]

model = utils.HF_SimpleModel(
    filenames,
    num_layers = 3,
    input_dim = 32,
    embed_dim = 64,
    output_dim = 32,
    num_heads = 4,
    num_freqs = 512,
    dim_feedforward = 128,
    input_dropout = 0.1,
    output_dropout = 0.0,
    dropout = 0.1,
    device = device
)
model = torch.compile(model, mode="max-autotune", fullgraph=True)

# Loss function and optimizer
loss_fn = torch.nn.MSELoss(reduction = "sum")
optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)

# Define schedulers
min_lr = 1e-7
k = min_lr / MAX_LR
linear_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=k, end_factor=1.0, total_iters=NUM_WARMUP_EPOCHS)

cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = (MAX_NUM_EPOCHS - NUM_WARMUP_EPOCHS))

# Combine them
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[linear_warmup, cosine_annealing],
    milestones=[NUM_WARMUP_EPOCHS]
)


train_losses = []
val_losses = []
min_test_loss = float('inf')

# train the model and record the loss for the training and validation datasets
for t in range(MAX_NUM_EPOCHS):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_losses.append(
        utils.train_loop([train_dataloader1, train_dataloader2], model, loss_fn, optimizer, scheduler, device)
    )
    validation_loss = utils.test_loop([val_dataloader], model, loss_fn, device)
    val_losses.append(validation_loss)

    if (validation_loss < min_test_loss):
        # save the model
        print("Saving model...")
        torch.save(
            model.state_dict(),
            "tMoTe2_2BPV_HF_model_num_freqs_512.pth",
        )
        print("Model saved!\n")
        min_test_loss = validation_loss
print("Done!\n")

print(f"Training Losses (1-{MAX_NUM_EPOCHS}): {train_losses}")
print(f"Test Losses (1-{MAX_NUM_EPOCHS}): {val_losses}\n")

utils.plot_losses(train_losses, val_losses)
