# imports
from utils import matrix_processing
import os

def main(L: int):
    # Set the environment variable to suppress the warning
    os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

    # get correct filename
    prefix = "/blue/yujiabin/awwab.azam/hartree-fock-code/misc_h5_files/small_t_Correlator_Nov"
    filename_small = f"{prefix}/RG_L{L}_FULL_from_reps.npz"
    filename_large = f"{prefix}/RG_L18_FULL_from_reps.npz"  # make sure to change this if needed

    # otherwise, make a new folder with the appropriate name
    print(
        f"\nStarting training for Richardson model on L={L}..."
    )
    
    # call the function
    matrix_processing(
        filename_small,
        filename_large,
        foldername=f"/blue/yujiabin/awwab.azam/hartree-fock-code/src/jobs/richardson_model_fixed/train_L{L}",
        repeat_trials=10,
        n_startup_trials=48,
        gpus_per_trial=0.5,
        total_steps=100,
        num_samples=200,
    )
    print(
        f"\nFinished training for Richardson model on L={L}!"
    )
    print("\nDone!")


if __name__ == '__main__':
    main(6)