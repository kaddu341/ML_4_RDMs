import torch
import utils
import json
from torch.utils.data import DataLoader
import h5py
import numpy as np
from ray.train import Checkpoint
from pathlib import Path
import pickle
import math

# functions to reformat output into a sequence of nxn matrices
# currently for tMoTe2 model
def reformat_nxn_matrix(array: np.ndarray):
    # input should be of shape (L^2, 2n^2)
    n_square = array.shape[-1] // 2
    # first, convert back to complex
    real_part = array[:, :n_square]
    imag_part = array[:, n_square:]
    array = real_part + (1j * imag_part)
    array = array.astype(np.complex128)

    # array should now be complex of shape (L^2, n^2)
    # need to reshape to (n, n, L^2)
    L_square = array.shape[0]
    n = int(math.sqrt(n_square))
    array = array.reshape(L_square, n, n)
    array = np.transpose(array)
    
    return array

def main_function(num_freqs):
    NUM_FREQS = num_freqs

    # change this when needed
    cutoff = None
    dim = 18
    # print(f"L is {dim}\n")
    indices = (5_000, 10_000)

    # directory for model + normalization constants
    folder_name = "tMoTe2_6x6_9x9_freq_test_lc_july5"
    dir = f"/blue/yujiabin/awwab.azam/hartree-fock-code/src/jobs/{folder_name}/"

    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # get the filenames
    dims = list(range(3, 19, 3))    # should go from 3 to 18 in increments of 3
    # filenames = [f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/BZindex_{d}x{d}.pt" for d in dims]
    filenames = [f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/true_momentum_{d}x{d}.pt" for d in dims]

    # load model (make sure matches what's in the folder)
    model = utils.HF_SimpleModel(
        filenames,
        num_layers = 3,
        input_dim = 32,
        embed_dim = 64,
        output_dim = 32,
        num_heads = 4,
        num_freqs = NUM_FREQS,
        dim_feedforward = 128,
        input_dropout = 0.0,
        output_dropout = 0.0,
        dropout = 0.0,
        device = device
    )

    # Load the saved state_dict
    print(f"Loading model from {dir}{NUM_FREQS}_test/tMoTe2_2BPV_HF_model_num_freqs_{NUM_FREQS}_final.pth")
    state_dict = torch.load(f"{dir}{NUM_FREQS}_test/tMoTe2_2BPV_HF_model_num_freqs_{NUM_FREQS}_final.pth")

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
    with open(dir + "normalization_constants.json", "r") as f:
        normalization_constants = json.load(f)

    x_mean = normalization_constants["x_mean"]
    x_std = normalization_constants["x_std"]
    y_mean = normalization_constants["y_mean"]
    y_std = normalization_constants["y_std"]

    print("Normalization Constants:")
    print(f"\nInputs Mean: {x_mean}, Inputs Standard Deviation: {x_std}")
    print(f"Targets Mean: {y_mean}, Targets Standard Deviation: {y_std}")

    # load dataset
    eval_dataset = utils.HF_3x3_Dataset(f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/tMoTe2_dataset_{dim}x{dim}_10K.h5", normalization_constants, indices=(5_000, 10_000))
    print(f"\nDataset size: {eval_dataset.__len__()}")

    # make a new DataLoader for testing purposes
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=100, shuffle=True, pin_memory=True, num_workers=2
    )

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    num_batches = len(eval_dataloader)
    test_loss = 0
    loss_fn = torch.nn.MSELoss()

    # instantiate a new file to write our results to
    file = h5py.File(f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/L_{dim}/output_{folder_name}_num_freqs_{NUM_FREQS}_for_test", 'w')
    count = 1

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in eval_dataloader:
            # Move data to the appropriate device (CPU/GPU)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # convert to correct format
            pred = utils.denormalize(pred, y_mean, y_std)
            pred = pred.cpu().numpy()
            for sequence in pred:
                result = reformat_nxn_matrix(sequence)    # need to change cutoff to deal w/ diff size
                result = np.stack([result.real, result.imag], axis=0)
                file.create_dataset(f"ML_{count}", data=result)
                count += 1

    test_loss /= num_batches
    print(f"Test loss: {test_loss:>8f} \n")

    # close file
    file.close()

if __name__ == "__main__":
    main_function(1024)
    main_function(512)
    main_function(128)