# A Mini-library of various functions and classes for the Hartree-Fock ML project

# imports
import json
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import h5py
import math

# DATA PREPROCESSING FUNCTIONS/UTILITIES



# functions to normalize/denormalize input & output (with masking if needed), respectively
def normalize(tensor, mean, std_dev):
    # now normalize and return the new tensor
    return (tensor - mean) / std_dev

def denormalize(tensor, mean, std_dev):
    # assuming no mask needed
    return (tensor * std_dev) + mean


def get_normalization_constants_3x3(filenames_dict, write_to="normalization_constants.json"):    # filenames_dict should be of form "filepath" : # of training samples
    stats = dict()
    with h5py.File(next(iter(filenames_dict.keys())), 'r') as f:
        num_indices = f['dataset'].shape[-1]
    
    for i in range(num_indices):
        stats[f"elt_{i}"] = {
            "initial": {},
            "results": {}
        }

        # first get the averages
        x_sum = 0.0
        y_sum = 0.0
        x_num_elements = 0
        y_num_elements = 0
        for filename in filenames_dict.keys():
            with h5py.File(filename, 'r') as f:
                num_samples = filenames_dict[filename]
                initial_data = f['dataset'][:num_samples, 0, :, i]
                results_data = f['dataset'][:num_samples, 1, :, i]

                x_sum += np.sum(initial_data)
                x_num_elements += initial_data.size

                y_sum += np.sum(results_data)
                y_num_elements += results_data.size
        
        x_mean = (x_sum / x_num_elements).item()
        y_mean = (y_sum / y_num_elements).item()
        stats[f"elt_{i}"]["initial"]["mean"] = x_mean
        stats[f"elt_{i}"]["results"]["mean"] = y_mean

        # now get the standard deviations
        x_variance = 0.0
        y_variance = 0.0
        for filename in filenames_dict.keys():
            with h5py.File(filename, 'r') as f:
                num_samples = filenames_dict[filename]
                initial_data = f['dataset'][:num_samples, 0, :, i]
                results_data = f['dataset'][:num_samples, 1, :, i]

                initial_data = initial_data - x_mean
                results_data = results_data - y_mean

                # square the result
                initial_data = initial_data ** 2
                results_data = results_data ** 2
                # take the sum
                x_variance += np.sum(initial_data)
                y_variance += np.sum(results_data)
                    
        # divide by num_elements - 1 to get the true variance
        x_variance = x_variance / (x_num_elements - 1)
        y_variance = y_variance / (y_num_elements - 1)
        # take the square root to get the standard deviation
        x_std = np.sqrt(x_variance).item()
        y_std = np.sqrt(y_variance).item()

        stats[f"elt_{i}"]["initial"]["std_dev"] = x_std
        stats[f"elt_{i}"]["results"]["std_dev"] = y_std

    # write normalization constants to a JSON file so that we can load them later
    with open(write_to, "w") as f:
        json.dump(stats, f)

    return stats


class HF_3x3_Dataset(Dataset):
    def __init__(self, filename, normalization_constants: dict = None, indices: Tuple[int, int] = None, transform = None, target_transform = None):
        # load the data
        with h5py.File(filename, 'r') as f:
            # get indices
            if indices is None:
                start_index = 0
                end_index = len(f['dataset'])
            else:
                start_index = indices[0]
                end_index = indices[1]

            self.initial_data = torch.from_numpy(f['dataset'][start_index : end_index, 0, :, :]).float()
            self.results_data = torch.from_numpy(f['dataset'][start_index : end_index, 1, :, :]).float()
        
        # load normalization constants from file
        if normalization_constants is None:
            with open("normalization_constants.json", "r") as f:
                stats = json.load(f)
        else:
            stats = normalization_constants
        
        # now normalize the data
        num_indices = self.initial_data.shape[-1]
        for i in range(num_indices):
            self.initial_data[..., i] = normalize(self.initial_data[..., i], stats[f"elt_{i}"]["initial"]["mean"], stats[f"elt_{i}"]["initial"]["std_dev"])
            self.results_data[..., i] = normalize(self.results_data[..., i], stats[f"elt_{i}"]["results"]["mean"], stats[f"elt_{i}"]["results"]["std_dev"])

        # other attributes
        self.transform = transform
        self.target_transform = target_transform
    
    # function to return length of dataset
    def __len__(self):
        return self.initial_data.shape[0]
    
    # function to return an (input, output) pair from the dataset
    def __getitem__(self, index):
        initial_arr = self.initial_data[index]
        results_arr = self.results_data[index]

        # apply any transforms as needed
        if self.transform:
            initial_arr = self.transform(initial_arr)
        if self.target_transform:
            results_arr = self.target_transform(results_arr)

        return initial_arr, results_arr


# NEURAL NETWORK FUNCTIONS/MODULES


def get_padding_mask(tensor: torch.Tensor):
    # Assuming tensor is of the form (batch_size, seq_length, dim_model) and the padding token is 0
    seq_length = tensor.shape[1]

    mask = (tensor.abs().sum(dim=-1) == 0).unsqueeze(1).repeat(1, seq_length, 1)
    return mask


def get_simple_coords(n: int) -> Tuple[List[int], List[int]]:
    x_coords = []
    y_coords = []
    for i in range(n):
        for j in range(n):
            x_coords.append(i)
            y_coords.append(j)
    return x_coords, y_coords


# POSITIONAL ENCODING METHODS
def get_bias_matrices(filenames, device=torch.device("cpu")):
    bias_dict = {}

    for filename in filenames:
        coords = torch.load(filename, weights_only=True)
        seq_len = coords.shape[0]

        # get x and y coordinates for each element of the sequence
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]

        # initialize bias matrix
        bias_matrix = torch.zeros((seq_len, seq_len, 4), device=device)
        # iterate over query indices (since this is a one-time thing we can afford to use loops instead of vectorization)
        for i in range(seq_len):
            # iterate over key indices
            for j in range(seq_len):
                bias_matrix[i, j] = torch.tensor(
                    [x_coords[i], y_coords[i], x_coords[j], y_coords[j]],
                    device=device,
                )

        # add to dictionary
        bias_dict[seq_len] = bias_matrix
    
    return bias_dict

# same as above but in OOP format
class StandardBiasGenerator:
    # constructor
    def __init__(self, filenames, device=torch.device("cpu")) -> None:
        self.bias_dict = {}

        for filename in filenames:
            coords = torch.load(filename, weights_only=True)
            seq_len = coords.shape[0]

            # get x and y coordinates for each element of the sequence
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]

            # initialize bias matrix
            bias_matrix = torch.zeros((seq_len, seq_len, 4), device=device)
            # iterate over query indices (since this is a one-time thing we can afford to use loops instead of vectorization)
            for i in range(seq_len):
                # iterate over key indices
                for j in range(seq_len):
                    bias_matrix[i, j] = torch.tensor(
                        [x_coords[i], y_coords[i], x_coords[j], y_coords[j]],
                        device=device,
                    )

            # add to dictionary
            self.bias_dict[seq_len] = bias_matrix

    def get_bias(self, src: torch.Tensor) -> torch.Tensor:
        # src should be of shape (batch_size, seq_len, dim_model)
        seq_len = src.shape[1]

        return self.bias_dict[seq_len]


# WEIGHT INITIALIZATION FUNCTIONS
def init_weights_truncated_normal(m, mean=0.0, std=0.02, trunc_std=2.0):
    """
    Initializes weights with a truncated normal distribution.
    The values are effectively drawn from N(mean, std)
    and re-sampled if they fall outside mean +/- trunc_std * std.
    """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=mean, std=std)
        while torch.any(m.weight < mean - trunc_std * std) or torch.any(m.weight > mean + trunc_std * std):
            # Re-sample weights that are outside the truncation range
            out_of_bounds = (m.weight < mean - trunc_std * std) | (m.weight > mean + trunc_std * std)
            new_weights = torch.normal(mean, std, size=m.weight.shape, device=m.weight.device)
            m.weight.data[out_of_bounds] = new_weights[out_of_bounds]
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weights_xavier(m, gain=1.0):
    """Initializes weights with Xavier (Glorot) uniform initialization."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weights_kaiming(m, nonlinearity='gelu'): # PyTorch's GELU is approximated as 'relu' for Kaiming
    """Initializes weights with Kaiming (He) normal initialization."""
    if isinstance(m, nn.Linear):
        # For GELU, 'relu' is a common approximation for Kaiming gain
        # If you were using strict ReLU, it would be 'relu'
        # If using LeakyReLU, it would be 'leaky_relu' and you'd specify 'a' parameter
        if nonlinearity == 'gelu': # GELU is similar to ReLU in terms of non-negative part
             nonlinearity_param = 'relu'
        else:
             nonlinearity_param = nonlinearity
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=nonlinearity_param)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# versions of f_theta to use
class f_theta_learnableFF(nn.Module):
    def __init__(self, num_freqs, device):
        super().__init__()
        self.num_freqs = num_freqs

        # matrix to hold the a_i's and b_i's (frequencies)
        self.W_r = nn.Linear(2, num_freqs, bias=False, device=device)

        # coefficient vector
        self.C = nn.Parameter(torch.ones(num_freqs, device=device))

        # power
        self.log_p = nn.Parameter(torch.tensor(0.0, device=device))

        self._initialize_weights()
        self.to(device)
    
    def _initialize_weights(self):
        # for the W_r matrix
        std_dev = 1.0
        nn.init.normal_(self.W_r.weight, mean=0.0, std=std_dev)
    
    def forward(self, bias_tensor):
        # compute bias matrix
        a_i_plus_b_i = self.W_r(bias_tensor)
        cosine_c = 1 - torch.cos(a_i_plus_b_i)
        C_square = self.C ** 2
        sigma = torch.sum(cosine_c * C_square, dim=-1)
        p = torch.exp(self.log_p)
        norm_p = torch.pow(sigma + 1e-6, p)
        B = torch.exp(-norm_p)
        
        return B

class ChatGPT_version_f_theta_lc(nn.Module):
    def __init__(self, num_freqs, device):
        super().__init__()
        self.num_freqs = num_freqs

        # frequency projection
        self.W_r = nn.Linear(2, num_freqs, bias=False, device=device)

        # learnable coefficient vector
        self.C = nn.Parameter(torch.ones(num_freqs, device=device))

        # log power
        self.log_p = nn.Parameter(torch.tensor(0.0, device=device))  # start with log(1)

        self._initialize_weights()
        self.to(device)

    def _initialize_weights(self):
        nn.init.normal_(self.W_r.weight, mean=0.0, std=1.0)

    def forward(self, bias_tensor):
        # bias_tensor: (batch_size, seq_len, seq_len, 4)
        k_1 = self.W_r(bias_tensor[..., 0:2])  # (B, S, S, num_freqs)
        k_2 = self.W_r(bias_tensor[..., 2:])   # (B, S, S, num_freqs)

        # combined feature
        phi_1 = torch.cos(k_1) + torch.sin(k_1)
        phi_2 = torch.cos(k_2) + torch.sin(k_2)
        difference = phi_1 - phi_2

        # weighted sum
        sigma = torch.sum(difference * self.C, dim=-1)
        sigma = torch.abs(sigma)

        # Clamp sigma to avoid extreme powers
        sigma = torch.clamp(sigma, min=1e-4, max=1e4)

        # Clamp log_p to avoid exploding p
        log_p_clamped = torch.clamp(self.log_p, min=-5.0, max=5.0)
        p = torch.exp(log_p_clamped)

        # decay factor and final bias
        decay_factor = torch.pow(sigma, p)
        decay_factor = torch.clamp(decay_factor, max=50.0)  # avoid exp(-inf)

        B = torch.exp(-decay_factor)

        return B

class f_theta_lc(nn.Module):
    def __init__(self, num_freqs, device):
        super().__init__()
        self.num_freqs = num_freqs

        # matrix to hold the a_i's and b_i's (frequencies)
        self.W_r = nn.Linear(2, num_freqs, bias=False, device=device)

        # coefficient vector
        self.C = nn.Parameter(torch.ones(num_freqs, device=device))

        # power
        self.log_p = nn.Parameter(torch.tensor(0.0, device=device))

        self._initialize_weights()
        self.to(device)
    
    def _initialize_weights(self):
        # for the W_r matrix
        std_dev = 1.0
        nn.init.normal_(self.W_r.weight, mean=0.0, std=std_dev)
    
    def forward(self, bias_tensor):
        # bias_tensor should have shape (seq_length, seq_length, 4)
        # where the last dimension contains (x_1, y_1, x_2, y_2)
        k_1 = self.W_r(bias_tensor[:, :, 0:2])
        k_2 = self.W_r(bias_tensor[:, :, 2:])
        difference = torch.cos(k_1) + torch.sin(k_1) - torch.cos(k_2) - torch.sin(k_2)
        sigma = torch.abs(torch.sum(difference * self.C, dim=-1))
        p = torch.exp(self.log_p)
        p = torch.clamp(p, min=0.5, max=3)  # <-- ADD THIS LINE
        decay_factor = torch.pow(sigma + 1e-6, p)
        B = torch.exp(-decay_factor)
        
        return B


# class to implement a single self-attention head
class Relative_AttentionHead(nn.Module):
    def __init__(
        self, dim_model: int, kdim: int, num_freqs: int = 70, device=torch.device("cpu")
    ) -> None:
        super().__init__()
        self.denominator = torch.sqrt(torch.tensor(float(kdim), device=device))

        self.f_theta = f_theta_lc(num_freqs=num_freqs, device=device)

        # initialize parameter matrices
        self.W_q = nn.Linear(dim_model, kdim, bias=False)
        self.W_k = nn.Linear(dim_model, kdim, bias=False)
        self.W_v = nn.Linear(dim_model, kdim, bias=False)

        # scaling parameter for SSMax function
        self.s = nn.Parameter(torch.tensor(1.0, device=device))

        self._initialize_weights()
        self.to(device)
    
    def _initialize_weights(self):
        init_weights_xavier(self.W_q)
        init_weights_xavier(self.W_k)
        init_weights_xavier(self.W_v)

    def forward(
        self,
        src: torch.Tensor,
        bias_tensor: torch.Tensor,
        multiplier: torch.Tensor,
    ) -> torch.Tensor:
        # Assuming src has shape (batch_size, seq_length, dim_model)
        # and bias_tensor has shape (seq_length, seq_length, 4)
        batch_size = src.shape[0]

        # compute bias matrix
        B = self.f_theta(bias_tensor)
        B = B.unsqueeze(0).expand(batch_size, -1, -1)

        # get Query, Key, and Value matrices for each sequence
        Q = self.W_q(src)
        K = self.W_k(src)
        V = self.W_v(src)

        # calculate attention logits
        K_T = torch.transpose(K, 1, 2)
        attn_logits = (torch.bmm(Q, K_T) + B) / self.denominator

        # multiply by (s * log n) for scalable softmax
        multiplier = self.s * multiplier
        attn_logits = attn_logits * multiplier

        # apply softmax to get final attention scores
        attn_weights = torch.nn.functional.softmax(attn_logits, dim=-1)
        attn_outputs = torch.bmm(attn_weights, V)
        return attn_outputs


# class to implement multi-head self-attention with relative positional encodings
class Relative_SelfAttention(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_freqs: int,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__()

        # make sure num_heads divides dim_model:
        assert dim_model % num_heads == 0, (
            "Number of heads must divide dimension of model"
        )

        # compute kdim = vdim
        kdim = dim_model // num_heads

        # initialize attention heads
        self.attention_heads = nn.ModuleList(
            [
                Relative_AttentionHead(dim_model=dim_model, kdim=kdim, num_freqs=num_freqs, device=device)
                for _ in range(num_heads)
            ]
        )

        # final linear layer
        if num_heads == 1:
            self.W_o = nn.Identity()
        else:
            self.W_o = nn.Linear(dim_model, dim_model)
        
        # Weight initialization
        self._initialize_weights()

        # move to device
        self.to(device)
    
    def _initialize_weights(self):
        init_weights_xavier(self.W_o)

    def forward(
        self,
        src: torch.Tensor,
        bias_tensor: torch.Tensor,
        multiplier: torch.Tensor,
    ) -> torch.Tensor:
        # src should have shape (batch_size, seq_length, dim_model)
        # pass src through the attention heads
        attn_results = [
            attn_head(src, bias_tensor, multiplier)
            for attn_head in self.attention_heads
        ]
        # concatenate results
        attn_results = torch.cat(attn_results, dim=-1)
        # pass through final linear layer
        return self.W_o(attn_results)


class HF_TransformerLayer(nn.Module):
    # initialize neural network layers
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_freqs: int,
        dim_feedforward: int,
        dropout: float,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.attention = Relative_SelfAttention(
            dim_model = embed_dim,
            num_heads = num_heads,
            num_freqs = num_freqs,
            device = device
        )
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self._initialize_weights()
        self.to(device)
    
    def _initialize_weights(self):
        for m in self.MLP.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        src: torch.Tensor,
        bias_tensor: torch.Tensor,
        multiplier: torch.Tensor,
    ) -> torch.Tensor:
        # run through attention block
        src2 = self.attention(src, bias_tensor, multiplier)
        src = self.norm1(src + self.dropout1(src2))

        # run through multilayer perceptron
        src2 = self.MLP(src)
        src = self.norm2(src + self.dropout2(src2))
        return src


class HF_SimpleModel(nn.Module):
    # initialize alternating attention/MLP layers
    def __init__(
        self,
        filenames,
        num_layers: int = 1,
        input_dim: int = 16,
        embed_dim: int = None,
        output_dim: int = None,
        num_heads: int = 1,
        num_freqs: int = 70,
        dim_feedforward: int = 128,
        input_dropout: float = 0.1,
        output_dropout: float = 0.0,
        dropout: float = 0.1,
        device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device

        # default behavior is to keep the same embedding dimension
        if embed_dim is None:
            embed_dim = input_dim
        if output_dim is None:
            output_dim = input_dim

        # embedding and output linear layers
        self.linear1 = nn.Linear(input_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, output_dim)

        # dropout layers
        self.dropout1 = nn.Dropout(input_dropout)
        self.dropout2 = nn.Dropout(output_dropout)

        # to get bias
        self.bias_dict = get_bias_matrices(filenames, device)

        self.layers = nn.ModuleList(
            [
                HF_TransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_freqs=num_freqs,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    device=device
                )
                for _ in range(num_layers)
            ]
        )
        # Weight Initialization
        self._initialize_weights()
        self.to(device)
    
    def _initialize_weights(self):
        init_weights_truncated_normal(self.linear1)
        init_weights_truncated_normal(self.linear2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should be of shape (batch_size, seq_len, input_dim)
        # get the bias matrix
        seq_len = x.shape[1]
        bias_tensor = self.bias_dict[seq_len]

        # compute log(n) where n = nonzero sequence length
        multiplier = torch.log(torch.tensor(seq_len, dtype=torch.float32, device=self.device))

        # run through model layers
        x = self.dropout1(self.linear1(x))
        for layer in self.layers:
            x = layer(
                src = x,
                bias_tensor = bias_tensor,
                multiplier = multiplier
            )
        x = self.dropout2(self.linear2(x))
        return x



# FUNCTIONS FOR TRAINING/OPTIMIZATION PROCEDURE


# functions for the training and testing loops
def train_loop(dataloaders: List[torch.utils.data.DataLoader], model, loss_fn, optimizer, scheduler, device):
    training_loss = 0.0
    num_batches = len(dataloaders[0])
    # make sure all the dataloaders have the same size
    for i, dataloader in enumerate(dataloaders):
        assert len(dataloader) == num_batches, f"Dataloader {i} has {len(dataloader)} batches, expected {num_batches}"

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    for batch in zip(*dataloaders):
        # each batch is of shape (X, y), (X, y), ...
        total_batch_loss = 0
        num_elements = 0

        for X, y in batch:
            # tensors should be of form (batch_size, seq_length, embed_dim)
            # Move data to the appropriate device (CPU/GPU)
            X, y = X.to(device), y.to(device)

            # Compute the predicted output
            pred = model(X)

            # compute the loss
            loss = loss_fn(pred, y)
            total_batch_loss += loss
            num_elements += torch.numel(y)
        
        total_batch_loss = total_batch_loss / num_elements
        training_loss += total_batch_loss.item()

        # Backpropagation
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # Gradient Clipping
        optimizer.step()
        optimizer.zero_grad()
    # Step the scheduler after every epoch
    scheduler.step()
    # Track learning rate
    print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")

    training_loss /= num_batches
    print(f"Train loss: {training_loss:>8f}")

    return training_loss

def test_loop(dataloaders: List[torch.utils.data.DataLoader], model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    num_batches = len(dataloaders[0])
    # make sure all the dataloaders have the same size
    for i, dataloader in enumerate(dataloaders):
        assert len(dataloader) == num_batches, f"Dataloader {i} has {len(dataloader)} batches, expected {num_batches}"
    test_loss = 0.0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch in zip(*dataloaders):
            total_batch_loss = 0.0
            num_elements = 0.0

            for X, y in batch:
                # Move data to the appropriate device (CPU/GPU)
                X, y = X.to(device), y.to(device)
                pred = model(X)
                total_batch_loss += loss_fn(pred, y).item()
                num_elements += torch.numel(y)
            
            total_batch_loss = total_batch_loss / num_elements
            test_loss += total_batch_loss

    test_loss /= num_batches
    print(f"Avg Test loss: {test_loss:>8f} \n")

    return test_loss


def plot_losses(train_losses, test_losses, img_filename = "train_test_losses.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, marker="s")  # Plot the list with markers
    plt.plot(test_losses, marker="o")  # Plot the list with markers
    plt.title("Train and Test Losses")
    plt.xlabel("Epoch No.")
    plt.ylabel("Value")
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(img_filename)  # Saves as PNG by default

    # Show the plot
    plt.show()
