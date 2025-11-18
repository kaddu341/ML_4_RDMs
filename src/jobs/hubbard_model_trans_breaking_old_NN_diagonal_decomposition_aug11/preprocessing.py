import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List
import torch
import os
from functools import partial
import math

MIN_ENERGY = -0.5655891666193353

def fold_OP_mat(OP_mat):
    # input should be of shape (2L^2, 2L^2)
    L = int(round(math.sqrt(OP_mat.shape[0] / 2)))
    print(f"L is {L}")

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

def get_diagonal_elements(folded_OP_mat):
    # input should be of shape (L**2, L**2, 2, 2)
    L = int(round(math.sqrt(folded_OP_mat.shape[0])))
    diagonals_list = []
    for i in range(L**2):
        y = (L**2) - i
        diagonal_piece = np.zeros((L, L, 2, 2))

        row_idx = 0
        while row_idx < y:
            k_1 = row_idx // L
            k_2 = row_idx % L
            offset = i + row_idx
            diagonal_piece[k_1, k_2, :, :] = folded_OP_mat[row_idx, offset, :, :]
            row_idx += 1

        diagonals_list.append(diagonal_piece)
    return diagonals_list

# helper function to convert an MxM matrix into a sequence of flattened vectors
def make_2x2_matrices(array: np.ndarray, diag_line=0) -> np.ndarray:
    # array should be of shape (2L^2, 2L^2)
    m_dim = 2

    # first we need to find L
    two_L_square = array.shape[0]
    L_square = two_L_square // m_dim
    L = int(round(math.sqrt(L_square)))

    # flatten alpha indices
    folded_array = fold_OP_mat(array)  # should be of shape (L^2, L^2, 2, 2)

    # get diagonal elements
    diag_list = get_diagonal_elements(folded_array)

    # get the specific diagonal we are looking for
    indices = []
    for i, diag_elts in enumerate(diag_list):
        if np.max(np.abs(diag_elts)) > 1e-4:
            indices.append(i)

    index = indices[diag_line]

    matrix = diag_list[index]  # matrix should be (L, L, 2, 2)
    matrix = matrix.reshape((L**2, 4))
    return matrix


# function to parse projected (translation-breaking) data
def parse_nxn_dataset_ndarray(io_filenames: Tuple[str, str]):
    # io_filenames should be a tuple of (input_filename, output_filename)
    input_filename, output_filename = io_filenames

    # add each group of matrices to the list
    with h5py.File(input_filename, "r") as in_file:
        # get number of samples
        num_samples = len(list(in_file.keys()))
        with h5py.File(output_filename, "r") as out_file:
            # now we can finally compute
            sequence_list = []
            for i in range(1, num_samples + 1):
                print(f"\rIndex: {i}", end='', flush=True)
                try:
                    # check if its in the lowest energy band
                    if (out_file[f'HF_energy_per_unit_cell_{i}'][()].real - MIN_ENERGY < 1e-4):
                        sequence_list.append(np.array(
                            [
                                make_2x2_matrices(np.transpose(in_file[f'P_initial_{i}'][()].conj())),
                                make_2x2_matrices(np.transpose(out_file[f'P_HF_result_{i}'][()].conj())),
                            ]
                        ))
                except (KeyError, OSError, IndexError) as e:
                    print(f"Error reading {i}: {e}")
                    continue
    
    sequence_list = np.array(sequence_list, dtype=np.float32)
    return sequence_list

# function to run preprocessing code in parallel
def parallelize_preprocessing(prefix: str, L: int, dataset_filename: str, num_samples = 2500, num_folders: int = 4):
    # make list of filenames to process
    args = []
    for i in range(1, num_folders + 1):
        full_path_prefix = f"{prefix}/L_{L}_U_1_number_of_random_initials_{num_samples}_occupied_{int((L**2)/2)}_{int((L**2)/2)}_index_{i}/"
        args.append((full_path_prefix + "P_HF_initial.h5", full_path_prefix + "P_HF_result.h5"))
    
    print("Filenames to process:")
    for i, (input_filename, output_filename) in enumerate(args):
        print(f"{i}: {input_filename} -> {output_filename}")
    print("Starting processing...")
    
    # run in parallel
    with ProcessPoolExecutor() as executor:
        results = executor.map(parse_nxn_dataset_ndarray, args)
    
    # concatenate the results
    results = np.concatenate(list(results), axis=0)

    # randomly shuffle the combined dataset
    rng = np.random.default_rng()
    rng.shuffle(results, axis=0)

    print(f"Final shape: {results.shape}")
    print(f"Data type: {results.dtype}")

    # save the results
    with h5py.File(f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/hubbard_model/{dataset_filename}", "w") as file:
        file.create_dataset(name="dataset", data=results, dtype=np.float32)
    print("Done!")


# This guard is CRUCIAL for multiprocessing to work correctly.
if __name__ == "__main__":
    # for L=8
    parallelize_preprocessing(
        prefix = "/blue/yujiabin/awwab.azam/hartree-fock-code/Julia_Hubbard_model_tunable_interaction_range_HF_For_ML_Trans_Breaking",
        L=8,
        dataset_filename="hubbard_8x8_lowest_energy_raw.h5",
        num_folders=4,
        num_samples=2500
    )

    # for L=10
    parallelize_preprocessing(
        prefix = "/blue/yujiabin/awwab.azam/hartree-fock-code/Julia_Hubbard_model_tunable_interaction_range_HF_For_ML_Trans_Breaking",
        L=10,
        dataset_filename="hubbard_10x10_lowest_energy_raw.h5",
        num_folders=4,
        num_samples=2500
    )

    # for L=18
    parallelize_preprocessing(
        prefix = "/blue/yujiabin/awwab.azam/hartree-fock-code/Julia_Hubbard_model_tunable_interaction_range_HF_For_ML_Trans_Breaking",
        L=18,
        dataset_filename="hubbard_18x18_lowest_energy_raw.h5",
        num_folders=1,
        num_samples=200
    )