# imports
import torch
import numpy as np
import utils
from torch.utils.data import DataLoader
import os
import h5py
import random
import sys

# state variables
U = 1
error_c='2e-5'
label = f"U_{U}_ground_state"

# for lr scheduler
MAX_LR = 1e-3
MAX_NUM_EPOCHS = 5_000   # 500 initially, then 5000
NUM_WARMUP_EPOCHS = 20

# Set the cache size limit to the largest possible integer
torch._dynamo.config.cache_size_limit = sys.maxsize

print(f"torch._dynamo.config.cache_size_limit is now set to: {torch._dynamo.config.cache_size_limit}")

torch.set_float32_matmul_precision('highest')

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

def training_script(num_freqs: int, index: int = 0, device = device):
    # Set random seed for reproducibility
    torch.manual_seed(index)
    np.random.seed(index)
    random.seed(index)

    # number of frequencies
    NUM_FREQS = num_freqs

    # make a new folder for this test
    folder_name = f"/blue/yujiabin/awwab.azam/hartree-fock-code/src/jobs/old_NN_U_{U}_ground_state/{NUM_FREQS}_test_index_{index}"

    # dataset files
    TRAIN_DATASET1 = f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/hubbard_model/diagonal_datasets/U_{U}_ground_state/{error_c}/hubbard_model_lowest_energy_8x8_U_{U}_ground_state.h5"
    TRAIN_DATASET2 = f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/hubbard_model/diagonal_datasets/U_{U}_ground_state/{error_c}/hubbard_model_lowest_energy_10x10_U_{U}_ground_state.h5"
    VAL_DATASET = f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/hubbard_model/diagonal_datasets/U_{U}_ground_state/{error_c}/hubbard_model_lowest_energy_18x18_U_{U}_ground_state.h5"

    filenames_dict = {
        TRAIN_DATASET1: 467,
        TRAIN_DATASET2: 1001,
    }
    chunks_8 = 20
    chunks_10 = 43

    # get normalization constants
    stats = utils.get_normalization_constants_3x3(filenames_dict, write_to=f"normalization_constants.json")

    print("Normalization Constants:")
    print(stats)

    # Check if CUDA is available, otherwise use CPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    dims = [8, 10, 18]    # could go (for example) from 8 to 18 in increments of 2
    filenames = [f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/hubbard_model/diagonal_datasets/diagonal_coords_{d}x{d}.pt" for d in dims]

    with h5py.File(TRAIN_DATASET1, 'r') as f:
        io_dim = f['dataset'].shape[-1]

    model = utils.HF_SimpleModel(
        filenames,
        num_layers = 3,
        input_dim = io_dim,
        embed_dim = 64,
        output_dim = io_dim,
        num_heads = 4,
        num_freqs = NUM_FREQS,
        dim_feedforward = 128,
        input_dropout = 0.1,
        output_dropout = 0.0,
        dropout = 0.1,
        device = device
    )

    try:
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' created successfully.\n")
        start_epoch = 0
    except FileExistsError:
        print(f"Folder '{folder_name}' already exists.\n")

        # Load the saved state_dict
        print(f"Loading model from {folder_name}/hubbard_model_2BPV_HF_model_num_freqs_{NUM_FREQS}_final.pth")
        state_dict = torch.load(f"{folder_name}/hubbard_model_2BPV_HF_model_num_freqs_{NUM_FREQS}_final.pth")

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

        # update start_epoch
        start_epoch = 500
        
    except OSError as e:
        print(f"Error creating folder: {e}\n")
    
    # compile the model
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    # make training and validation datasets
    training_data1 = utils.HF_3x3_Dataset(TRAIN_DATASET1, stats)
    print(f"Training dataset length: {training_data1.__len__()}\n")

    training_data2 = utils.HF_3x3_Dataset(TRAIN_DATASET2, stats)
    print(f"Training dataset length: {training_data2.__len__()}\n")

    validation_data = utils.HF_3x3_Dataset(VAL_DATASET, stats)
    print(f"Test dataset length: {validation_data.__len__()}\n")

    # training and test dataloaders
    train_dataloader1 = DataLoader(training_data1, chunks_8, shuffle=True, pin_memory=True, num_workers=1)
    train_dataloader2 = DataLoader(training_data2, chunks_10, shuffle=True, pin_memory=True, num_workers=1)
    val_dataloader = DataLoader(validation_data, 4, shuffle=True, pin_memory=True, num_workers=1)

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

    # train the model and record the loss for the training and validation datasets
    for t in range(MAX_NUM_EPOCHS):
        current_epoch = start_epoch + t + 1
        print(f"Epoch {current_epoch}\n-------------------------------")
        train_losses.append(
            utils.train_loop([train_dataloader1, train_dataloader2], model, loss_fn, optimizer, scheduler, device)
        )
        val_losses.append(utils.test_loop([val_dataloader], model, loss_fn, device))
        # save the model occasionally
        if (t % 100 == 0):
            print("Saving model...")
            torch.save(
                model.state_dict(),
                f"{folder_name}/hubbard_model_2BPV_HF_model_num_freqs_{NUM_FREQS}_epoch_{current_epoch}.pth",
            )
            print("Model saved!\n")
    print("Done!\n")

    print(f"Training Losses (1-{MAX_NUM_EPOCHS}): {train_losses}")
    print(f"Test Losses (1-{MAX_NUM_EPOCHS}): {val_losses}\n")

    utils.plot_losses(train_losses, val_losses, img_filename=f"{folder_name}/train_test_losses.png")

    # saving model
    torch.save(
        model.state_dict(),
        f"{folder_name}/hubbard_model_2BPV_HF_model_num_freqs_{NUM_FREQS}_final.pth",
    )

    return min(val_losses)

# MAIN program
min_loss = training_script(48, index=52, device=device)
print(f"Minimum Loss: {min_loss}")