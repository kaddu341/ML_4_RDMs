# imports
import torch
from torch import nn
import utils
from torch.utils.data import DataLoader
from pathlib import Path
import tempfile
from functools import partial
from ray import tune
from ray.tune import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import json
import numpy as np

TRAIN_DATASET1 = "/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/tMoTe2_dataset_6x6_10K.h5"
TRAIN_DATASET2 = "/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/tMoTe2_dataset_9x9_10K.h5"
VAL_DATASET = "/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/tMoTe2_dataset_12x12_10K.h5"

def get_param_count(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}\n")
    return total_params

def training_script(config, normalization_constants, max_num_epochs, batch_size, device):
    dims = list(range(3, 19, 3))    # should go from 3 to 18 in increments of 3
    filenames = [f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/BZindex_{d}x{d}.pt" for d in dims]

    bias_generator = utils.StandardBiasGenerator(filenames, device)
    model = utils.HF_SimpleModel(
        bias_generator = bias_generator,
        num_layers = config["num_layers"],
        input_dim = 32,
        embed_dim = config["embed_dim"],
        output_dim = 32,
        num_heads = config["num_heads"],
        hidden_size = config["hidden_size"],
        num_freqs = config["num_freqs"],
        dim_feedforward = config["dim_feedforward"],
        input_dropout = config["dropout"],
        output_dropout = 0.0,
        dropout = config["dropout"],
        device = device,
    )

    # check to see if we can use multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("Using nn.DataParallel!\n")

    # Loss function and optimizer
    loss_fn = torch.nn.MSELoss(reduction = "sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["max_lr"])

    # loading information from checkpoint if possible
    checkpoint = get_checkpoint()
    if checkpoint:
        # if checkpoint exists:
        with checkpoint.as_directory() as checkpoint_dir:
            # get path to checkpoint directory
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                # load state
                checkpoint_state = pickle.load(fp)
            # set variables
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            scheduler_state = checkpoint_state["scheduler_state_dict"]
    else:
        # otherwise if checkpoint doesn't exist, set to default
        start_epoch = 0
        scheduler_state = None
        # initialize weights
        # utils.initialize_weights(model)
    
    # Define schedulers
    min_lr = 1e-7
    k = min_lr / config["max_lr"]
    linear_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=k, end_factor=1.0, total_iters=config["num_warmup_epochs"])

    cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = (max_num_epochs - config["num_warmup_epochs"]))

    # Combine them
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[linear_warmup, cosine_annealing],
        milestones=[config["num_warmup_epochs"]]
    )
    # Corrected version
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    # make training and validation datasets
    # build training, validation?, and test datasets
    training_data1 = utils.HF_3x3_Dataset(TRAIN_DATASET1, normalization_constants)
    print(f"Training dataset length: {training_data1.__len__()}\n")

    training_data2 = utils.HF_3x3_Dataset(TRAIN_DATASET2, normalization_constants)
    print(f"Training dataset length: {training_data2.__len__()}\n")

    validation_data = utils.HF_3x3_Dataset(VAL_DATASET, normalization_constants, indices=(0, 5_000))
    print(f"Test dataset length: {validation_data.__len__()}\n")

    # training and test dataloaders
    train_dataloader1 = DataLoader(training_data1, batch_size, shuffle=True, pin_memory=True, num_workers=1)
    train_dataloader2 = DataLoader(training_data2, batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_dataloader = DataLoader(validation_data, batch_size, shuffle=True, pin_memory=True, num_workers=2)

    # loop over the dataset
    total_epochs = max_num_epochs
    for t in range(start_epoch, total_epochs):
        # train the model
        utils.train_loop([train_dataloader1, train_dataloader2], model, loss_fn, optimizer, scheduler, device)

        # send validation loss to Ray Tune
        validation_loss = utils.test_loop([val_dataloader], model, loss_fn, device)
        checkpoint_data = {
            "epoch": t,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),  # <-- Add this
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            tune.report(
                {"loss": validation_loss},
                checkpoint=checkpoint,
            )

    print("Finished Training")


def main(num_samples: int, max_num_epochs: int):
    # declare filenames (replaced by global const)
    # dataset_file1 = "/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/..."
    # dataset_file2 = "/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/..."


    filenames_dict = {
        TRAIN_DATASET1: 10_000,
        TRAIN_DATASET2: 10_000,
    }

    # get normalization constants
    x_mean, x_std, y_mean, y_std = utils.get_normalization_constants_3x3(filenames_dict)

    print("Normalization Constants:")
    print(f"\nInputs Mean: {x_mean}, Inputs Standard Deviation: {x_std}")
    print(f"Targets Mean: {y_mean}, Targets Standard Deviation: {y_std}")

    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # get norm constants dict
    with open("normalization_constants.json", "r") as f: 
        normalization_constants = json.load(f)
    
    # set batch size
    batch_size = 100

    # Run fine-tuning code
    # Define Ray Tune search space and setup everything
    config = {
        "num_layers": tune.grid_search([3]),
        "embed_dim": tune.grid_search([64]),
        "num_heads": tune.grid_search([4]),
        "hidden_size": tune.grid_search([32]),
        "num_freqs": tune.grid_search([100, 200, 400, 800, 1600]),
        "dim_feedforward": tune.grid_search([128]),
        "dropout": tune.grid_search([0.1]),

        "max_lr": tune.grid_search([1e-3]),
        "num_warmup_epochs": tune.grid_search([20]),
    }
    raytune_scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=40,    # was originally 30
            reduction_factor=2,
        )
    result = tune.run(
            run_or_experiment=partial(
                training_script, 
                normalization_constants = normalization_constants,
                max_num_epochs = max_num_epochs, 
                batch_size = batch_size,
                device = device
            ),
            resources_per_trial={"cpu": 3, "gpu": 1},
            config=config,
            num_samples=1,      # change to =num_samples if using tune.choice() or ~1 if using tune.grid_search()
            storage_path="/blue/yujiabin/awwab.azam/hartree-fock-code/src/ray_results",
            scheduler=raytune_scheduler
        )

    # find out best combination of hyperparameters so far
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}\n")

    # now get the filenames again
    dims = list(range(3, 19, 3))    # should go from 3 to 18 in increments of 3
    filenames = [f"/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/tMoTe2_model/BZindex_{d}x{d}.pt" for d in dims]

    # get best-performing model
    bias_generator = utils.StandardBiasGenerator(filenames, device)
    best_trained_model = utils.HF_SimpleModel(
        bias_generator = bias_generator,
        num_layers = best_trial.config["num_layers"],
        input_dim = 32,
        embed_dim = best_trial.config["embed_dim"],
        output_dim = 32,
        num_heads = best_trial.config["num_heads"],
        hidden_size = best_trial.config["hidden_size"],
        num_freqs = best_trial.config["num_freqs"],
        dim_feedforward = best_trial.config["dim_feedforward"],
        input_dropout = best_trial.config["dropout"],
        output_dropout = 0.0,
        dropout = best_trial.config["dropout"],
        device = device,
    )

    # print the number of trainable parameters in the model.
    get_param_count(best_trained_model)
    
    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="loss", mode="min")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["model_state_dict"])

    # saving model
    torch.save(
        best_trained_model.state_dict(),
        "tMoTe2_2BPV_HF_model.pth",
    )
    print("Model saved to file!\n")

    # Evaluate loss on test dataset
    print("Evaluating model on test dataset...")
    test_data = utils.HF_3x3_Dataset(VAL_DATASET, normalization_constants, indices=(5_000, 10_000))
    print(f"Test dataset length: {test_data.__len__()}\n")

    test_dataloader = DataLoader(
        test_data, batch_size, shuffle=True, pin_memory=True, num_workers=2
    )
    utils.test_loop([test_dataloader], best_trained_model, torch.nn.MSELoss(reduction = "sum"), device)


if __name__ == "__main__":
    num_samples = 100
    max_num_epochs = 500

    # run main function
    main(num_samples, max_num_epochs)