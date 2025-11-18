import json
import math
import random
import tempfile
from functools import partial
from itertools import product
from pathlib import Path
from typing import Literal

# import h5py
import matplotlib.pyplot as plt
import numpy as np
import ray.cloudpickle as pickle
import torch
from optuna.samplers import TPESampler
from ray import tune
from ray.tune import Checkpoint, get_checkpoint
from ray.tune.search import Repeater
from ray.tune.search.optuna import OptunaSearch
from scipy.interpolate import RegularGridInterpolator
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset


def plot_complex_matrix(
    matrix,
    title="OP matrix",
    interpolation="none",
    vmin=-1.0,
    vmax=1.0,
    log_scale=False,
):
    """
    Plots a matrix. If the matrix is complex, plots both the real and imaginary parts.
    If the matrix is real, plots only the real part.

    Args:
        matrix (np.ndarray): Matrix to plot.
        log_scale (bool): Whether to use log scale for color mapping. Default is False.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    def get_plot_data(data):
        if log_scale:
            # Add a small epsilon to avoid log(0)
            epsilon = 1e-12
            return np.log10(np.abs(data) + epsilon)
        else:
            return data

    if np.iscomplexobj(matrix):
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        im0 = axs[0].imshow(
            get_plot_data(matrix.real),
            cmap="RdBu",
            interpolation=interpolation,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im0, ax=axs[0])
        axs[0].set_title(f"Re of {title}" + (" (log scale)" if log_scale else ""))

        im1 = axs[1].imshow(
            get_plot_data(matrix.imag),
            cmap="BrBG",
            interpolation=interpolation,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im1, ax=axs[1])
        axs[1].set_title(f"Im of {title}" + (" (log scale)" if log_scale else ""))

        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(10, 8))
        plt.imshow(
            get_plot_data(matrix),
            cmap="RdBu",
            interpolation=interpolation,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar()
        plt.title(f"Re of {title}" + (" (log scale)" if log_scale else ""))
        plt.tight_layout()
        plt.show()


def get_V_coords(L):
    V = np.zeros((L, L, 2))
    for k_1 in range(L):
        for k_2 in range(L):
            V[k_1, k_2] = np.array([k_1 / L, k_2 / L])
    return V


def scatter_plot(matrix, title="Real Part of OP"):
    # input should be of shape (L, L, 2, 2)
    L = matrix.shape[0]
    V = get_V_coords(L)
    X = V[:, :, 1]
    Y = V[:, :, 0]

    figure, axes = plt.subplots(2, 2, figsize=(12, 12))

    scatter = None

    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            ax.set_title(f"({i + 1}, {j + 1})")
            scatter = ax.scatter(
                X, Y, c=matrix[:, :, i, j], cmap="plasma", vmin=-1.0, vmax=1.0
            )
            ax.set_aspect("equal")
            ax.set_facecolor("w")

    # general settings for the whole figure
    figure.tight_layout(
        rect=[0, 0, 0.9, 1]
    )  # Adjust layout to make space for the colorbar
    figure.suptitle(title, y=1.03, fontsize=25)

    # --- New: Create a single colorbar for the entire figure ---
    # Create a new axes for the colorbar on the right side of the figure
    cbar_ax = figure.add_axes([0.92, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    figure.colorbar(scatter, cax=cbar_ax)

    # Show plot
    plt.show()


def fold_OP_mat(OP_mat):
    # input should be of shape (2L^2, 2L^2)
    L = int(round(math.sqrt(OP_mat.shape[0] / 2)))
    # print(f"L is {L}")

    # fold the alpha's (2x2 matrices)
    folded_OP = np.zeros((L**2, L**2, 2, 2))
    real_part = OP_mat.real

    for k_1 in range(L):
        for k_2 in range(L):
            for k_1_prime in range(L):
                for k_2_prime in range(L):
                    row_idx = (2 * k_1 * L) + (2 * k_2)
                    col_idx = (2 * k_1_prime * L) + (2 * k_2_prime)
                    alpha_matrix = real_part[
                        row_idx : row_idx + 2, col_idx : col_idx + 2
                    ]
                    i = k_1 * L + k_2
                    j = k_1_prime * L + k_2_prime
                    folded_OP[i, j, :, :] = alpha_matrix

    # return the result
    return folded_OP


def unfold_OP_mat(folded_OP_mat):
    # input of shape (L^2, L^2, 2, 2)
    L = int(round(math.sqrt(folded_OP_mat.shape[0])))
    # print(f"L is {L}")

    OP_mat = np.zeros((2 * (L**2), 2 * (L**2)), dtype=np.complex128)

    for k_1 in range(L):
        for k_2 in range(L):
            for k_1_prime in range(L):
                for k_2_prime in range(L):
                    alpha_matrix = folded_OP_mat[
                        k_1 * L + k_2, k_1_prime * L + k_2_prime, :, :
                    ]
                    for i in range(2):
                        for j in range(2):
                            alpha_1 = i
                            alpha_2 = j
                            row_idx = (2 * k_1 * L) + (2 * k_2) + alpha_1
                            col_idx = (2 * k_1_prime * L) + (2 * k_2_prime) + alpha_2
                            OP_mat[row_idx, col_idx] = alpha_matrix[i, j] + (1j * 0)

    # return the result
    return OP_mat


def get_diagonal_line(folded_OP_mat, relation):
    # folded_OP_mat should be of shape (L^2, L^2, 2, 2)
    L = int(round(math.sqrt(folded_OP_mat.shape[0])))
    diagonal_matrix = np.zeros((L, L, 2, 2))

    # loop over OP
    for v_1_L in range(L):
        for v_2_L in range(L):
            v_1 = v_1_L / L
            v_2 = v_2_L / L
            v_1_prime = relation(v_1)
            v_2_prime = relation(v_2)

            v_1_prime_L = int(round(v_1_prime * L))
            v_2_prime_L = int(round(v_2_prime * L))

            # get row and column indices
            row_idx = (v_1_L * L) + v_2_L
            col_idx = (v_1_prime_L * L) + v_2_prime_L

            # fill in the diagonal matrix
            diagonal_matrix[v_1_L, v_2_L, :, :] = folded_OP_mat[row_idx, col_idx, :, :]

    return diagonal_matrix


def reformat_diagonal(diagonal_matrix, relation):
    # diagonal_matrix should be of shape (L, L, 2, 2)
    L = diagonal_matrix.shape[0]
    folded_OP_mat = np.zeros((L**2, L**2, 2, 2))

    # loop over the diagonal matrix
    for v_1_L in range(L):
        for v_2_L in range(L):
            v_1 = v_1_L / L
            v_2 = v_2_L / L
            v_1_prime = relation(v_1)
            v_2_prime = relation(v_2)

            v_1_prime_L = int(round(v_1_prime * L))
            v_2_prime_L = int(round(v_2_prime * L))

            # get row and column indices
            row_idx = (v_1_L * L) + v_2_L
            col_idx = (v_1_prime_L * L) + v_2_prime_L

            # fill in the folded OP matrix
            folded_OP_mat[row_idx, col_idx, :, :] = diagonal_matrix[v_1_L, v_2_L, :, :]

    return folded_OP_mat


def extract_diagonals(folded_OP_mat):
    return (
        get_diagonal_line(folded_OP_mat, relation=lambda v: v),
        get_diagonal_line(folded_OP_mat, relation=lambda v: (v + 0.5) % 1),
    )


def reformat_matrix_from_diagonals(diagonal_matrices):
    folded_OP_mat = reformat_diagonal(diagonal_matrices[0], relation=lambda v: v)
    folded_OP_mat += reformat_diagonal(
        diagonal_matrices[1], relation=lambda v: (v + 0.5) % 1
    )
    return folded_OP_mat


def get_interpolated_matrix(OP_mat, L_target: int = 18, method: str = "pchip"):
    # OP_mat should be of shape (2L^2, 2L^2)
    folded_OP_mat = fold_OP_mat(OP_mat)
    diag_list = extract_diagonals(folded_OP_mat)
    L = diag_list[0].shape[0]

    V_LxL = get_V_coords(L_target)

    test_list = []

    for diagonal_matrix in diag_list:
        diagonal_piece = np.zeros((L + 1, L + 1, 2, 2))
        diagonal_piece[:L, :L, :, :] = diagonal_matrix
        diagonal_piece[L, :L, :, :] = diagonal_matrix[0, :, :, :]
        diagonal_piece[:L, L, :, :] = diagonal_matrix[:, 0, :, :]
        v_L = np.array([i / L for i in range(L + 1)])

        interpolated_piece = np.zeros((L_target, L_target, 2, 2))
        for j in range(2):
            interpolator = RegularGridInterpolator(
                (v_L, v_L),
                diagonal_piece[:, :, j, j],
                method=method,
                bounds_error=False,
                fill_value=None,
            )
            pred = interpolator(V_LxL)
            interpolated_piece[:, :, j, j] = pred

        test_list.append(interpolated_piece)

    test_list = tuple(test_list)
    test_OP = reformat_matrix_from_diagonals(test_list)
    test_OP = unfold_OP_mat(test_OP)
    return test_OP


# Define the index function. Note: Python uses 0-based indexing.
# This function takes 1-based inputs like the Mathematica version
# but returns a 0-based index for use with NumPy arrays.
def index(ik1, ik2, s, L):
    """Maps 1-based (ik1, ik2, s) to a 0-based linear index."""
    return (ik1 - 1) * L * 2 + (ik2 - 1) * 2 + (s - 1)


def calculate_nspinpolarized_table(s, P, L):
    """
    Calculates the spin-polarized density table for a given spin 's'.
    This function implements the vectorized version of the Mathematica 'Sum'.
    """
    # 1. Extract the L^2 x L^2 submatrix for the given spin 's' from P.
    k_indices_1_based = list(product(range(1, L + 1), repeat=2))

    linear_indices = [index(ik1, ik2, s, L) for ik1, ik2 in k_indices_1_based]

    # np.ix_ allows for efficient extraction of submatrices
    P_s = P[np.ix_(linear_indices, linear_indices)]

    # 2. Create the Fourier matrix E, where E_kr = exp(i * k . R)
    k_coords = np.arange(L)
    R_coords = np.arange(L)

    # Create L^2 k-vectors and R-vectors
    k_grid_x, k_grid_y = np.meshgrid(k_coords, k_coords, indexing="ij")
    R_grid_x, R_grid_y = np.meshgrid(k_coords, k_coords, indexing="ij")

    k_vectors = np.stack([k_grid_x.ravel(), k_grid_y.ravel()], axis=1)
    R_vectors = np.stack([R_grid_x.ravel(), R_grid_y.ravel()], axis=1)

    # Calculate k . R for all combinations
    k_dot_R = np.einsum("bi,ci->bc", k_vectors, R_vectors)  # shape: (L^2, L^2)

    E = np.exp(1j * 2 * np.pi / L * k_dot_R)

    # 3. Calculate the density n(R,s) using the vectorized formula.
    # n_s = diag(E^H @ P_s^* @ E) / L^2
    # @ is the matrix multiplication operator
    n_table_flat = np.diag(E.conj().T @ P_s.conj() @ E) / (L**2)

    # 4. Reshape the flat array of L^2 values into an L x L matrix
    # The result should be real, so we take the real part to discard
    # any small imaginary noise from floating point inaccuracies.
    return n_table_flat.real.reshape(L, L)


def plot_quantity(P, L):
    # Calculate the spin-up and spin-down density tables
    print("Calculating spin-up density table (nupTab)...")
    nupTab = calculate_nspinpolarized_table(1, P, L)  # s=1 for spin up

    print("Calculating spin-down density table (ndownTab)...")
    ndownTab = calculate_nspinpolarized_table(2, P, L)  # s=2 for spin down

    print("Calculations complete. Now plotting.")

    # Calculate the difference (magnetization)
    magnetization = nupTab - ndownTab

    # Plot the result using Matplotlib, which is the equivalent of MatrixPlot
    fig, axes = plt.subplots(1, 4, figsize=(12, 12))
    ax = axes[0]
    im = ax.matshow(magnetization, vmin=-1.5, vmax=1.5)
    ax.set_title("Spin Polarization ($n_{up} - n_{down}$)")
    ax.tick_params(axis="both", which="major", labelsize=8)
    # ax.set_xlabel("$R_2$")
    # ax.set_ylabel("$R_1$")

    ax = axes[1]
    ax.matshow(nupTab, vmin=-1.5, vmax=1.5)
    ax.set_title("$n_{up}$")
    ax.tick_params(axis="both", which="major", labelsize=8)
    # ax.set_xlabel("$R_2$")
    # ax.set_ylabel("$R_1$")

    ax = axes[2]
    ax.matshow(ndownTab, vmin=-1.5, vmax=1.5)
    ax.set_title("$n_{down}$")
    ax.tick_params(axis="both", which="major", labelsize=8)
    # ax.set_xlabel("$R_2$")
    # ax.set_ylabel("$R_1$")

    ax = axes[3]
    ax.matshow(nupTab + ndownTab, vmin=-1.5, vmax=1.5)
    ax.set_title("$n_{up} + n_{down}$")
    ax.tick_params(axis="both", which="major", labelsize=8)
    # ax.set_xlabel("$R_2$")
    # ax.set_ylabel("$R_1$")

    # Add a color bar, equivalent to PlotLegends -> Automatic
    cbar_ax = fig.add_axes([0.92, 0.35, 0.03, 0.3])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    plt.show()


def check_D4_symmetries(matrix, rtol=1e-5, atol=1e-8):
    """
    Check which D4 (dihedral group of order 8) symmetries a square matrix preserves.

    D4 consists of:
        - Rotations: 0°, 90°, 180°, 270°
        - Reflections: horizontal, vertical, main diagonal, anti-diagonal

    Args:
        matrix (np.ndarray): Square matrix to check.
        rtol (float): Relative tolerance for floating point comparisons.
        atol (float): Absolute tolerance for floating point comparisons.

    Returns:
        dict: Mapping symmetry names -> True/False.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square to check D4 symmetries.")

    symmetries = {}

    # --- Rotations ---
    symmetries["rotation_0"] = True  # identity, always preserved
    symmetries["rotation_90"] = np.allclose(
        matrix, np.rot90(matrix, 1), rtol=rtol, atol=atol
    )
    symmetries["rotation_180"] = np.allclose(
        matrix, np.rot90(matrix, 2), rtol=rtol, atol=atol
    )
    symmetries["rotation_270"] = np.allclose(
        matrix, np.rot90(matrix, 3), rtol=rtol, atol=atol
    )

    # --- Reflections ---
    symmetries["reflection_vertical"] = np.allclose(
        matrix, np.fliplr(matrix), rtol=rtol, atol=atol
    )
    symmetries["reflection_horizontal"] = np.allclose(
        matrix, np.flipud(matrix), rtol=rtol, atol=atol
    )
    symmetries["reflection_main_diag"] = np.allclose(
        matrix, matrix.T, rtol=rtol, atol=atol
    )
    symmetries["reflection_anti_diag"] = np.allclose(
        matrix, np.flipud(np.fliplr(matrix)).T, rtol=rtol, atol=atol
    )

    # Print results
    print("D4 Symmetry check results:")
    for sym, val in symmetries.items():
        print(f"  {sym:25}: {'Yes' if val else 'No'}")

    return symmetries


def enforce_D4_symmetries_pytorch(
    tensor_batch: torch.Tensor, symmetries: dict = None
) -> torch.Tensor:
    """
    Enforce chosen D4 symmetries (rotations/reflections) on a batch of square tensors
    by averaging over the specified transformations.

    Args:
        tensor_batch (torch.Tensor): Batch of square tensors of shape (B, H, W).
        symmetries (dict): Dictionary to specify which symmetries to enforce.
            Possible values:
                - "rotation_0"   (identity, always preserved)
                - "rotation_90"
                - "rotation_180"
                - "rotation_270"
                - "reflection_vertical"
                - "reflection_horizontal"
                - "reflection_main_diag"
                - "reflection_anti_diag"

    Returns:
        torch.Tensor: Symmetrized batch, shape (B, H, W).
    """
    if tensor_batch.shape[-2] != tensor_batch.shape[-1]:
        raise ValueError("The last two dimensions of the input tensor must be square.")

    # Handle Default case
    if symmetries is None:
        symmetries = {
            "rotation_0": True,
            "rotation_90": True,
            "rotation_180": True,
            "rotation_270": True,
            "reflection_vertical": True,
            "reflection_horizontal": True,
            "reflection_main_diag": True,
            "reflection_anti_diag": True,
        }

    transforms = []

    # loop over the dictionary
    for symmetry, val in symmetries.items():
        # apply symmetry only if you're supposed to
        if val is True:
            match symmetry:
                case "rotation_0":
                    # R0 should always be true
                    transforms.append(tensor_batch)
                case "rotation_90":
                    transforms.append(torch.rot90(tensor_batch, k=1, dims=(-2, -1)))
                case "rotation_180":
                    transforms.append(torch.rot90(tensor_batch, k=2, dims=(-2, -1)))
                case "rotation_270":
                    transforms.append(torch.rot90(tensor_batch, k=3, dims=(-2, -1)))
                case "reflection_vertical":
                    transforms.append(
                        torch.flip(tensor_batch, dims=(-1,))
                    )  # flip last dim
                case "reflection_horizontal":
                    transforms.append(
                        torch.flip(tensor_batch, dims=(-2,))
                    )  # flip second-to-last dim
                case "reflection_main_diag":
                    transforms.append(tensor_batch.transpose(-2, -1))
                case "reflection_anti_diag":
                    transforms.append(
                        torch.rot90(tensor_batch.transpose(-2, -1), k=2, dims=(-2, -1))
                    )
                case _:
                    raise ValueError(f"Unknown symmetry: {symmetry}")

    # Stack & average
    symmetrized = torch.mean(torch.stack(transforms, dim=0), dim=0)
    return symmetrized


def get_normalized_coords(L: int):
    # should go from -1 to 1
    v_L = (np.array([i / L for i in range(L + 1)]) - 0.5) * 2

    V_1, V_2 = np.meshgrid(v_L, v_L, indexing="ij")

    # combine them into one array
    V = np.stack([V_1, V_2], axis=-1)
    return V


def normalize(tensor: torch.Tensor | np.ndarray, mean: float, std_dev: float):
    return (tensor - mean) / std_dev


def denormalize(tensor: torch.Tensor | np.ndarray, mean: float, std_dev: float):
    return (tensor * std_dev) + mean


class OP_Diagonal(Dataset):
    def __init__(self, diagonal_matrix: torch.Tensor, mean: float, std_dev: float):
        # diagonal_matrix should be of shape (L+1, L+1)
        L = diagonal_matrix.shape[0] - 1

        # standardize dataset
        self.diagonal_matrix = normalize(diagonal_matrix, mean=mean, std_dev=std_dev)

        # get coordinates
        self.coords = torch.from_numpy(get_normalized_coords(L)).float()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index > 0:
            raise IndexError

        return self.coords, self.diagonal_matrix


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords


def apply_periodic_boundary_conditions(
    diagonal_matrix: np.ndarray, mode: Literal["opposite", "adjacent"] = "opposite"
) -> torch.Tensor:
    # diagonal_matrix should be of shape (L, L)
    L = diagonal_matrix.shape[0]

    # apply periodic boundary conditions
    temporary_matrix = np.zeros((L + 1, L + 1))
    temporary_matrix[:L, :L] = diagonal_matrix
    temporary_matrix[L, L] = diagonal_matrix[0, 0]

    if mode == "opposite":
        temporary_matrix[L, :L] = diagonal_matrix[0, :]
        temporary_matrix[:L, L] = diagonal_matrix[:, 0]
    elif mode == "adjacent":
        # fill in the antidiagonal corners
        temporary_matrix[0, L] = diagonal_matrix[0, 0]
        temporary_matrix[L, 0] = diagonal_matrix[0, 0]
        # now the sides
        temporary_matrix[L, 1:L] = np.flip(diagonal_matrix[1:, 0])
        temporary_matrix[1:L, L] = np.flip(diagonal_matrix[0, 1:])
    else:
        raise ValueError("Invalid value for mode")

    # convert to PyTorch tensor
    temporary_matrix = torch.from_numpy(temporary_matrix).float()
    return temporary_matrix


def hessian(y, x):
    """hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)
    """
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(
        meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]
    ).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y, create_graph=True)[0][
                ..., :
            ]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.0
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][
            ..., i : i + 1
        ]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def jacobian(y, x):
    """jacobian of y wrt x"""
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(
        y.device
    )  # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[..., i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status


def TV_prior(
    k1: float, model: nn.Module, coords: torch.Tensor, device=torch.device("cpu")
):
    # coords should be of shape (batch_size, L, L, 2)
    # sample a random set of points from the domain
    coords_rand = 2 * (
        torch.rand(
            (
                coords.shape[0],  # batch size
                (coords.shape[1] * coords.shape[2]) // 2,  # No. of sampled pts
                coords.shape[3],  # dimension
            )
        ).to(device)
        - 0.5
    )

    # evaluate the model at those points
    rand_output, rand_input = model(coords_rand)

    # compute the loss
    prior_loss = k1 * torch.abs(gradient(rand_output, rand_input)).mean()
    return prior_loss


def FH_prior(
    k1: float, model: nn.Module, coords: torch.Tensor, device=torch.device("cpu")
):
    # coords should be of shape (batch_size, L, L, 2)
    # sample a random set of points from the domain
    coords_rand = 2 * (
        torch.rand(
            (
                coords.shape[0],  # batch size
                (coords.shape[1] * coords.shape[2]) // 2,  # No. of sampled pts
                coords.shape[3],  # dimension
            )
        ).to(device)
        - 0.5
    )

    # evaluate the model at those points
    rand_output, rand_input = model(coords_rand)

    img_hessian, status = hessian(rand_output, rand_input)
    img_hessian = img_hessian.view(*img_hessian.shape[0:2], -1)
    hessian_norm = img_hessian.norm(dim=-1, keepdim=True)

    prior_loss = k1 * torch.abs(hessian_norm).mean()
    return prior_loss


def run_model_inference(
    model: nn.Module,
    X: torch.Tensor,
    mean: float,
    std_dev: float,
    symmetries: dict = None,
    downsampler=None,
    N: int = 1,
    plot=True,
):
    model.eval()
    with torch.no_grad():
        if downsampler is not None:
            X = X.unsqueeze(-2).expand(-1, -1, N, -1)
            blurred_coords = downsampler(X)  # blurred_coords should be (L, L, N, 2)
            eval_output, _ = model(blurred_coords)
            eval_output = torch.mean(eval_output, dim=-2)
        else:
            eval_output, _ = model(X)

        eval_output = enforce_D4_symmetries_pytorch(eval_output.squeeze(-1), symmetries)
        eval_output = eval_output.squeeze()

        eval_output = denormalize(eval_output, mean, std_dev).cpu().numpy()

    if plot is True:
        plot_complex_matrix(eval_output, "NN prediction")
    return eval_output


def print_comparison(actual, pred):
    max_diff = np.max(np.abs(actual - pred))
    mse_loss = np.mean(np.abs(actual - pred) ** 2)
    print(f"Max diff: {max_diff}")
    print(f"MSE: {mse_loss}")
    return max_diff, mse_loss


def training_script(
    config,
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_epochs: int,
    symmetries: dict,
):
    # Set random seed for reproducibility
    seed = config[tune.search.repeater.TRIAL_INDEX]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # instantiate model(s)
    model = Siren(
        in_features=2,
        hidden_features=config["hidden_size"],
        hidden_layers=3,
        out_features=1,
        outermost_linear=True,
        first_omega_0=6,
        hidden_omega_0=30,
    )

    # Check if CUDA is available, otherwise use CPU
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
    loss_fn = nn.MSELoss()

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    # make dataloaders
    trainloader = DataLoader(
        train_dataset, batch_size=1, num_workers=1, pin_memory=True
    )  # set pin_memory and num_workers for GPU
    valloader = DataLoader(test_dataset, batch_size=1, num_workers=1, pin_memory=True)

    # get inputs and outputs for train and test
    X_train, Y_train = next(iter(trainloader))
    X_train, Y_train = X_train.to(device), Y_train.to(device)

    X_test, Y_test = next(iter(valloader))
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    resolution_coords = (
        torch.from_numpy(get_normalized_coords(config["downsampling_resolution"]))
        .float()
        .unsqueeze(0)
        .to(device)
    )

    for epoch in range(start_epoch, num_epochs):
        # first get training loss
        model.train()
        model_output, _ = model(resolution_coords)
        model_output = enforce_D4_symmetries_pytorch(
            model_output.squeeze(-1), symmetries
        )
        model_output = torch.nn.functional.interpolate(
            model_output.unsqueeze(0),
            size=(9, 9),
            mode=config["interp_mode"],
            align_corners=True,
            antialias=False,
        ).squeeze(0)

        Y_pred, _ = model(X_train)
        Y_pred = Y_pred.squeeze(-1)

        training_loss = loss_fn(model_output, Y_train) + loss_fn(Y_pred, Y_train)

        if config["prior_type"] == "TV":
            training_loss = training_loss + TV_prior(1e-5, model, X_test, device)
        elif config["prior_type"] == "FH":
            training_loss = training_loss + FH_prior(1e-7, model, X_test, device)
        elif config["prior_type"] is not None:
            raise ValueError("Invalid option")

        # backpropagation
        optimizer.zero_grad()
        training_loss.backward()
        # Clip gradients by norm
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # lr_scheduler.step()

        # and now for test
        model.eval()
        with torch.no_grad():
            test_output, _ = model(X_test)
            test_output = enforce_D4_symmetries_pytorch(
                test_output.squeeze(-1), symmetries
            )
            test_loss = loss_fn(test_output, Y_test).item()
            max_diff = torch.max(torch.abs(Y_test - test_output)).item()

        if (epoch + 1) % 10 == 0:  # checkpoint every 10 epochs
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                tune.report(
                    {"mse_loss": test_loss, "max_diff": max_diff},
                    checkpoint=checkpoint,
                )
        else:
            # report metrics without checkpoint
            tune.report({"mse_loss": test_loss, "max_diff": max_diff})

    print("\nFinished Training (for this trial)!")
    print(f"Final Max Diff: {max_diff}")
    print(f"Final MSE Loss: {test_loss}")


def matrix_processing(
    OP_8x8: np.ndarray,
    OP_18x18: np.ndarray,
    foldername: str = "U_1_ground_state/diag_1_matrix_11",
    repeat_trials: int = 20,  # how many times to repeat each configuration
    n_startup_trials: int = 25,  # how many points to sample before TPE kicks in
    gpus_per_trial: int = 0,  # pretty self-explanatory
    total_steps: int = 100,  # number of epochs
    num_samples: int = 200,  # how many different configurations to try
):
    # determine which type of B.C.'s to apply
    print("Checking clipped symmetries:")
    clipped_OP_8x8 = OP_8x8[1:, 1:]
    clipped_OP_18x18 = OP_18x18[1:, 1:]
    if (check_D4_symmetries(clipped_OP_8x8)["reflection_anti_diag"] is True) and (
        check_D4_symmetries(clipped_OP_18x18)["reflection_anti_diag"] is True
    ):
        mode = "adjacent"
    else:
        mode = "opposite"
    print(f"Selected mode: {mode}")

    # preprocess the diagonal matrices into PyTorch tensors
    print("\nPreprocessing matrices...")
    diag_matrix_8x8 = apply_periodic_boundary_conditions(OP_8x8, mode=mode)
    diag_matrix_18x18 = apply_periodic_boundary_conditions(OP_18x18, mode=mode)

    # get the NumPy equivalents
    diag_matrix_8x8_np = diag_matrix_8x8.numpy()
    diag_matrix_18x18_np = diag_matrix_18x18.numpy()

    # plot them to visualize what's going on
    plot_complex_matrix(
        diag_matrix_8x8_np, title="(1, 1) elt of diagonal 8x8 OP matrix"
    )
    plot_complex_matrix(
        diag_matrix_18x18_np, title="(1, 1) elt of diagonal 18x18 OP matrix"
    )

    print("\nSymmetries for 8x8 Diagonal:")
    symmetries_8x8 = check_D4_symmetries(diag_matrix_8x8_np)
    print("\nSymmetries for 18x18 Diagonal:")
    symmetries_18x18 = check_D4_symmetries(diag_matrix_18x18_np)
    # make sure they're the same
    assert symmetries_8x8 == symmetries_18x18, (
        "Symmetries are not the same for 8x8 and 18x18"
    )

    mean = torch.mean(diag_matrix_8x8).item()
    std_dev = torch.std(diag_matrix_8x8).item()

    print(f"\nMean: {mean}, Standard Deviation: {std_dev}")

    stats = {
        "mean": mean,
        "std_dev": std_dev,
    }

    with open(f"{foldername}/normalization_constants.json", "w") as f:
        json.dump(stats, f)

    # make datasets for the first diagonal
    diag_dataset_8x8 = OP_Diagonal(diag_matrix_8x8, mean=mean, std_dev=std_dev)
    diag_dataset_18x18 = OP_Diagonal(diag_matrix_18x18, mean=mean, std_dev=std_dev)
    plot_complex_matrix(
        diag_dataset_8x8.__getitem__(0)[1].numpy(),
        title="Standardized Diagonal for 8x8",
    )
    plot_complex_matrix(
        diag_dataset_18x18.__getitem__(0)[1].numpy(),
        title="Standardized Diagonal for 18x18",
    )

    print("\nBeginning hyperparameter optimization...")

    # configure search space
    config = {
        "hidden_size": tune.qrandint(256, 800, q=2),
        "downsampling_resolution": tune.choice([16, 18, 24, 32]),
        "interp_mode": tune.choice(["bilinear", "bicubic"]),
        "prior_type": tune.choice(["FH", "TV", None]),
    }
    optuna_search = OptunaSearch(
        sampler=TPESampler(multivariate=True, n_startup_trials=n_startup_trials),
        metric="max_diff",
        mode="min",
    )
    repeater = Repeater(
        optuna_search,
        repeat=repeat_trials,
        set_index=True,
    )
    tuner = tune.Tuner(
        tune.with_resources(
            partial(
                training_script,
                train_dataset=diag_dataset_8x8,
                test_dataset=diag_dataset_18x18,
                num_epochs=total_steps,
                symmetries=symmetries_18x18,
            ),
            resources={"cpu": 1, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            metric="max_diff",
            mode="min",
            search_alg=repeater,
            num_samples=(num_samples * repeat_trials),  # budget for trials
        ),
        param_space=config,
        run_config=tune.RunConfig(
            storage_path="/blue/yujiabin/awwab.azam/hartree-fock-code/src/ray_results",
        )
    )

    results = tuner.fit()
    print("\nRay Tune finished.")

    # print results!
    best_result = results.get_best_result(
        metric="max_diff", mode="min"
    )  # , scope="all"
    print("\nBest config:", best_result.config)
    print("Max diff:", best_result.metrics["max_diff"])
    print("MSE Loss:", best_result.metrics["mse_loss"])

    print("\nEvaluating results...")

    # get the best model
    best_trained_model = Siren(
        in_features=2,
        hidden_features=best_result.config["hidden_size"],
        hidden_layers=3,
        out_features=1,
        outermost_linear=True,
        first_omega_0=6,
        hidden_omega_0=30,
    )

    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_trained_model.to(device)
    print(f"Using device: {device}\n")

    best_checkpoint = best_result.get_best_checkpoint(metric="max_diff", mode="min")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_state = best_checkpoint_data["model_state_dict"]
        best_trained_model.load_state_dict(best_state)

        # test accuracy w.r.t. original distribution
        X_18 = torch.from_numpy(get_normalized_coords(18)).float().to(device)
        output_18x18 = run_model_inference(
            best_trained_model,
            X_18,
            mean=mean,
            std_dev=std_dev,
            symmetries=symmetries_18x18,
        )
        print_comparison(diag_matrix_18x18_np, output_18x18)

    # save model files to disk
    torch.save(best_state, f"{foldername}/model_state_dict.pth")
    torch.save(best_trained_model, f"{foldername}/model.pth")
    print("\nSaved model files to disk!")