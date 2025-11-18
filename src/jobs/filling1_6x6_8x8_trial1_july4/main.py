import torch
import utils
import json
from torch.utils.data import DataLoader

# filling number (1 or 2)
fi = 1

# directory for model + normalization constants
folder_name = "filling1_6x6_8x8_trial1_july4"
dir = f"/blue/yujiabin/awwab.azam/hartree-fock-code/src/jobs/{folder_name}/"

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load model (make sure matches what's in the folder)
model = utils.HF_SimpleModel(
    num_layers=3, input_dim=32, num_heads=2, dim_feedforward=256, dropout=0.0, device=device
)
model.load_state_dict(
    torch.load(
        dir + "tunable_interaction_range3_6x6_8x8_model.pth",
        weights_only=True,
    )
)

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

for L in [9, 10, 12, 14, 15, 16, 18, 22, 24, 25, 26, 28, 32, 34, 35, 36, 38, 42, 44, 45, 46, 48]:
    # load dataset
    eval_dataset = torch.load(f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tunable_interaction_range_datasets3/filling_1/test_{L}x{L}.pt", weights_only=True)
    print(f"\nDataset shape: {eval_dataset.shape}")

    # normalize dataset using the same constants
    eval_dataset[:, 0, :, :] = utils.normalize(eval_dataset[:, 0, :, :], x_mean, x_std)
    eval_dataset[:, 1, :, :] = utils.normalize(eval_dataset[:, 1, :, :], y_mean, y_std)

    # make a new dataset and DataLoader for testing purposes
    eval_dataset = utils.HartreeFockDataset(eval_dataset, type="full")
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=3
    )

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    num_batches = len(eval_dataloader)
    test_loss = 0
    loss_fn = torch.nn.MSELoss()

    tensor_list = []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in eval_dataloader:
            # Move data to the appropriate device (CPU/GPU)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            tensor_list.append(pred)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test loss: {test_loss:>8f} \n")

    result = torch.cat(tensor_list, dim=0)
    print(f"Final Result Shape: {result.shape}")

    # Denormalize Output
    # We only need to use the target normalization constants this time!
    result = utils.denormalize(result, y_mean, y_std)

    torch.save(result, f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/results/output_{folder_name}_{L}x{L}.pt")