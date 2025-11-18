# imports
import h5py
import numpy as np
import scienceplots
import itertools
from pathlib import Path
import os

from utils import (
    extract_diagonals,
    fold_OP_mat,
    matrix_processing,
    get_interpolated_matrix,
)

# state variables
U = 2
error_c='2e-5'
label = f"U_{U}_ground_state"

def main():
    # Set the environment variable to suppress the warning
    os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

    with h5py.File(
        f"/blue/yujiabin/awwab.azam/hartree-fock-code/Julia_Hubbard_model_tunable_interaction_range_HF_For_ML_Trans_Breaking/{label}/train_{error_c}/{label}.h5",
        "r",
    ) as f:
        OP_8x8 = f["OP_8x8"][()]
        print(OP_8x8.shape)
        OP_18x18 = f["OP_18x18"][()]
        print(OP_18x18.shape)

    testing_mode = False  # change this to False when running "for real"

    # first make sure they have the same pattern
    assert np.max(np.abs(get_interpolated_matrix(OP_8x8, 18) - OP_18x18)) < 0.5, (
        "Sorry, patterns do not match"
    )

    for diag_idx, alpha_1, alpha_2 in itertools.product((1, 0), repeat=3):  # reverse order for parallel acceleration
        diagonal_matrix_8x8 = extract_diagonals(fold_OP_mat(OP_8x8))[diag_idx][
            :, :, alpha_1, alpha_2
        ]
        diagonal_matrix_18x18 = extract_diagonals(fold_OP_mat(OP_18x18))[diag_idx][
            :, :, alpha_1, alpha_2
        ]

        # check to make sure it has significantly nonzero entries
        if np.max(np.abs(diagonal_matrix_18x18)) < 0.1:
            print(
                f"Elt ({alpha_1 + 1}, {alpha_2 + 1}) of Diagonal {diag_idx + 1} skipped: No significantly nonzero entries."
            )
            continue

        # otherwise, make a new folder with the appropriate name
        print(
            f"\nStarting training for ({alpha_1 + 1}, {alpha_2 + 1}) element of Diagonal {diag_idx + 1}..."
        )
        dir_name = f"diag_{diag_idx + 1}_matrix_{alpha_1 + 1}{alpha_2 + 1}"
        path = Path(dir_name)
        path.mkdir(
            exist_ok=testing_mode
        )  # can throw an error if the folder already exists, depends on testing_mode

        # call the function
        matrix_processing(
            OP_8x8=diagonal_matrix_8x8,
            OP_18x18=diagonal_matrix_18x18,
            foldername=str(path.resolve()),
            repeat_trials=20,
            n_startup_trials=48,
            gpus_per_trial=0.1,
            total_steps=100,
            num_samples=200,
        )
        print(
            f"\nFinished training for ({alpha_1 + 1}, {alpha_2 + 1}) element of Diagonal {diag_idx + 1}!"
        )
    print("\nDone!")


if __name__ == '__main__':
    main()