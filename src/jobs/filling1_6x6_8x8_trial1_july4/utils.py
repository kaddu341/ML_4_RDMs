# A Mini-library of various functions and classes for the Hartree-Fock ML project

# imports
import json
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

# DATA PREPROCESSING FUNCTIONS/UTILITIES


# function to combine datasets and add padding tokens
def combine_datasets(tensor_list: list):
    # find max sequence length
    max_len = 0
    for tensor in tensor_list:
        max_len = max(max_len, tensor.shape[2])

    # pad all tensors to max_len
    for i, tensor in enumerate(tensor_list):
        padding = (0, 0, 0, max_len - tensor.shape[2])
        tensor_list[i] = torch.nn.functional.pad(tensor, padding)

    # combine tensors
    combined_tensor = torch.cat(tensor_list, 0)

    # shuffle the combined tensor to randomize the order
    combined_tensor = combined_tensor[torch.randperm(combined_tensor.shape[0])]

    return combined_tensor


# function to get a mask where 1 denotes valid tokens and 0 denotes invalid ones
def get_reverse_mask(tensor: torch.Tensor) -> torch.Tensor:
    nonzero_indices = (tensor.abs().sum(dim=-1) != 0).float()
    return nonzero_indices.unsqueeze(-1).tile((tensor.shape[-1],))


# function to get the masked mean and standard deviation of a tensor (assuming padding token is 0)
def get_stats(tensor: torch.Tensor):
    # first, we need a mask because of the padding tokens
    reverse_mask = get_reverse_mask(tensor)

    # now get the mean of the tensor
    mean = tensor.sum() / reverse_mask.sum()

    # now get the standard deviation
    variance = ((tensor - (mean * reverse_mask)) ** 2).sum() / (reverse_mask.sum() - 1)
    std_dev = torch.sqrt(variance)

    return mean.item(), std_dev.item()


# functions to normalize/denormalize input & output (with masking if needed), respectively
def normalize(tensor, mean, std_dev):
    # first get the mask to handle 0 padding tokens, if there are any
    reverse_mask = get_reverse_mask(tensor)
    # now normalize and return the new tensor
    return (tensor - (mean * reverse_mask)) / std_dev

def denormalize(tensor, mean, std_dev):
    # assuming no mask needed
    return (tensor * std_dev) + mean


# function to transform input and target variables to have zero mean and unit variance and store the normalization constants
def normalize_dataset(
    dataset: torch.Tensor, train_percentage=0.8, filename="normalization_constants.json"
):
    # slice to get training set
    cutoff = round(train_percentage * len(dataset))
    train_split = dataset[0:cutoff]

    # now get statistics
    inputs, outputs = train_split[:, 0, :, :], train_split[:, 1, :, :]
    stats = dict()
    stats["x_mean"], stats["x_std"] = get_stats(inputs)
    stats["y_mean"], stats["y_std"] = get_stats(outputs)
    print(f"Input Mean: {stats['x_mean']}, Input Standard Deviation: {stats['x_std']}")
    print(
        f"Targets Mean: {stats['y_mean']}, Targets Standard Deviation: {stats['y_std']}"
    )

    # normalize the dataset
    dataset[:, 0, :, :] = normalize(
        dataset[:, 0, :, :], stats["x_mean"], stats["x_std"]
    )
    dataset[:, 1, :, :] = normalize(
        dataset[:, 1, :, :], stats["y_mean"], stats["y_std"]
    )

    # write normalization constants to a JSON file so that we can load them later
    with open(filename, "w") as f:
        json.dump(stats, f)

    return dataset


# custom dataset class
class HartreeFockDataset(Dataset):
    def __init__(
        self,
        dataset: torch.Tensor,
        type: str,
        transform=None,
        target_transform=None,
        split: Tuple[float, float] = (0.8, 0.1),
    ):
        # select indices to partition the dataset
        cutoff1 = round(split[0] * len(dataset))
        cutoff2 = round((split[0] + split[1]) * len(dataset))

        # choose which dataset to use
        match type:
            case "full":
                self.data = dataset
            case "train":
                self.data = dataset[:cutoff1]
            case "validation":
                self.data = dataset[cutoff1:cutoff2]
            case "test":
                self.data = dataset[cutoff2:]
            case _:
                print(
                    "Invalid output; must be one of 'full', 'train', 'validation', or 'test'"
                )
                return

        # other attributes
        self.transform = transform
        self.target_transform = target_transform

    # function to return length of dataset
    def __len__(self):
        return self.data.size(0)

    # function to return an (input, output) pair from the dataset
    def __getitem__(self, index):
        initial_matrix = self.data[index][0]
        results_matrix = self.data[index][1]

        # apply any transforms as needed
        if self.transform:
            initial_matrix = self.transform(initial_matrix)
        if self.target_transform:
            results_matrix = self.target_transform(results_matrix)

        return initial_matrix, results_matrix


# NEURAL NETWORK FUNCTIONS/MODULES


def get_padding_mask(tensor: torch.Tensor):
    # Assuming tensor is of the form (batch_size, seq_length, dim_model) and the padding token is 0
    seq_length = tensor.shape[1]

    mask = (tensor.sum(dim=-1) == 0).unsqueeze(1).repeat(1, seq_length, 1)
    return mask


def get_simple_coords(n: int) -> Tuple[List[int], List[int]]:
    x_coords = []
    y_coords = []
    for i in range(n):
        for j in range(n):
            x_coords.append(i)
            y_coords.append(j)
    return x_coords, y_coords


class BiasGenerator:
    # constructor
    def __init__(self, device=torch.device("cpu")) -> None:
        self.device = device
        self.bias_dict = {}

    def generate_bias_matrix(self, seq_len: int) -> torch.Tensor:
        # first, get L
        L = round(np.sqrt(seq_len))

        # get x and y coordinates for each element of the sequence
        x_coords, y_coords = get_simple_coords(L)

        # initialize bias matrix
        bias_matrix = torch.zeros((seq_len, seq_len, 2), device=self.device)
        # iterate over query indices (since this is a one-time thing we can afford to use loops instead of vectorization)
        for i in range(seq_len):
            # iterate over key indices
            for j in range(seq_len):
                bias_matrix[i, j] = torch.tensor(
                    [(x_coords[i] - x_coords[j]) / L, (y_coords[i] - y_coords[j]) / L],
                    device=self.device,
                )

        # add to dictionary
        self.bias_dict[seq_len] = bias_matrix
        # return
        return bias_matrix

    def get_bias(self, src: torch.Tensor) -> torch.Tensor:
        # src should be of shape (batch_size, padded_seq_len, dim_model)
        batch_size, padded_seq_len = src.shape[0:2]

        # calculate sequence lengths for each batch element
        seq_lens = (src.sum(dim=-1) != 0).sum(dim=1).tolist()

        # initialize bias
        bias = torch.zeros(
            (batch_size, padded_seq_len, padded_seq_len, 2), device=self.device
        )

        for index, seq_len in enumerate(seq_lens):
            # check if seq_len is in the dictionary
            if seq_len in self.bias_dict:
                bias_matrix = self.bias_dict[seq_len]
            else:
                bias_matrix = self.generate_bias_matrix(seq_len)

            # fill in the bias tensor
            bias[index, :seq_len, :seq_len] = bias_matrix

        return bias


# class to implement a single self-attention head
class Relative_AttentionHead(nn.Module):
    def __init__(
        self, dim_model: int, kdim: int, hidden_size: int, device=torch.device("cpu")
    ) -> None:
        super().__init__()
        self.denominator = torch.sqrt(torch.tensor(float(kdim), device=device))

        # initialize parameter matrices
        self.W_q = nn.Linear(dim_model, kdim, bias=False)
        self.W_k = nn.Linear(dim_model, kdim, bias=False)
        self.W_v = nn.Linear(dim_model, kdim, bias=False)

        # scaling parameter for SSMax function
        self.s = nn.Parameter(torch.tensor(1.0, device=device))

        # initialize learnable continuous function
        self.f_theta = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.to(device)

    def forward(
        self,
        src: torch.Tensor,
        bias_tensor: torch.Tensor,
        padding_mask: torch.Tensor,
        multiplier: torch.Tensor,
    ) -> torch.Tensor:
        # Assuming src has shape (batch_size, seq_length, dim_model)
        # compute bias matrix
        B = self.f_theta(bias_tensor).squeeze(-1)

        # get Query, Key, and Value matrices for each sequence
        Q = self.W_q(src)
        K = self.W_k(src)
        V = self.W_v(src)

        # calculate attention logits
        K_T = torch.transpose(K, 1, 2)
        attn_logits = (torch.bmm(Q, K_T) + B) / self.denominator

        # multiply by (s * log n) for scalable softmax
        multiplier = (self.s * multiplier).view(-1, 1, 1)
        attn_logits = attn_logits * multiplier

        # attention masking for padding tokens
        attn_logits.masked_fill_(padding_mask, float("-inf"))

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
        hidden_size: int,
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
                Relative_AttentionHead(dim_model, kdim, hidden_size, device)
                for _ in range(num_heads)
            ]
        )

        # final linear layer
        self.W_o = nn.Linear(dim_model, dim_model, bias=False)

        # move to device
        self.to(device)

    def forward(
        self,
        src: torch.Tensor,
        bias_tensor: torch.Tensor,
        padding_mask: torch.Tensor,
        multiplier: torch.Tensor,
    ) -> torch.Tensor:
        # src should have shape (batch_size, seq_length, dim_model)
        # pass src through the attention heads
        attn_results = [
            attn_head(src, bias_tensor, padding_mask, multiplier)
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
        hidden_size: int,
        dim_feedforward: int,
        dropout: float,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.attention = Relative_SelfAttention(
            embed_dim, num_heads, hidden_size, device
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
        self.to(device)

    def forward(
        self,
        src: torch.Tensor,
        bias_tensor: torch.Tensor,
        padding_mask: torch.Tensor,
        multiplier: torch.Tensor,
    ) -> torch.Tensor:
        # run through attention block
        src2 = self.attention(src, bias_tensor, padding_mask, multiplier)
        src = self.norm1(src + self.dropout1(src2))

        # run through multilayer perceptron
        src2 = self.MLP(src)
        src = self.norm2(src + self.dropout2(src2))
        return src


class HF_SimpleModel(nn.Module):
    # initialize alternating attention/MLP layers
    def __init__(
        self,
        num_layers: int = 1,
        input_dim: int = 16,
        embed_dim: int = None,
        num_heads: int = 1,
        hidden_size: int = 32,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__()

        # default behavior is to keep the same embedding dimension
        if embed_dim is None:
            embed_dim = input_dim

        # embedding and output linear layers
        self.linear1 = nn.Linear(input_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, input_dim)

        # to get bias
        self.bias_generator = BiasGenerator(device)

        self.layers = nn.ModuleList(
            [
                HF_TransformerLayer(
                    embed_dim, num_heads, hidden_size, dim_feedforward, dropout, device
                )
                for _ in range(num_layers)
            ]
        )
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get the bias tensor
        bias_tensor = self.bias_generator.get_bias(x)

        # get the mask for padding tokens
        padding_mask = get_padding_mask(x)

        # compute log(n) where n = nonzero sequence length
        N = (x.sum(dim=-1) != 0).sum(-1)
        multiplier = torch.log(N.float())

        # run through model layers
        x = self.linear1(x)
        for layer in self.layers:
            x = layer(x, bias_tensor, padding_mask, multiplier)
        x = self.linear2(x)
        return x


# FUNCTIONS FOR TRAINING/OPTIMIZATION PROCEDURE


def masked_MSE_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # first: get mask to ignore padding tokens
    reverse_mask = get_reverse_mask(target)
    # calculate loss
    L = (input - target) ** 2
    # ignore padding tokens
    L = L * reverse_mask
    # get average
    mse_loss = L.sum() / reverse_mask.sum()
    return mse_loss


# functions for the training and testing loops
def train_loop(dataloader, model, loss_fn, optimizer, device):
    training_loss = 0
    num_batches = len(dataloader)

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    for X, y in dataloader:
        # tensors should be of form (batch_size, seq_length, embed_dim)
        # Move data to the appropriate device (CPU/GPU)
        X, y = X.to(device), y.to(device)

        # Compute the predicted output
        pred = model(X)

        # compute the loss
        loss = loss_fn(pred, y)

        training_loss += loss.item()

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # Gradient Clipping
        optimizer.step()
        optimizer.zero_grad()

    training_loss /= num_batches
    print(f"Train loss: {training_loss:>8f}")

    return training_loss


def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            # Move data to the appropriate device (CPU/GPU)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test loss: {test_loss:>8f} \n")

    return test_loss


def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, marker="s")  # Plot the list with markers
    plt.plot(test_losses, marker="o")  # Plot the list with markers
    plt.title("Train and Test Losses")
    plt.xlabel("Epoch No.")
    plt.ylabel("Value")
    plt.grid(True)

    # Save the plot to a file
    plt.savefig("train_test_losses.png")  # Saves as PNG by default

    # Show the plot
    plt.show()
