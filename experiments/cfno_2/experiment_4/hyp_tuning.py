2  # train_one_model.py
import os

from autocvd import autocvd

autocvd(num_gpus=1)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import sys
from torch.nn import MSELoss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from dataloader.dataloader_3d import dataset_sr
from training.training_cnn import training_model
from experiments.cfno_2.experiment_4.fno_2_exp_2 import FNO_2

# from datetime import datetime
import gc
import optuna
from optuna import trial
import multiprocessing as mp

# This brings 'src' into the path
from typing import Any, Literal, Optional
import subprocess

config_file = yaml.safe_load(open("config.yaml", "r"))

study_name = "cfno_hyp_tuning"
config_file["training"]["epochs"] = 80
config_file["training"]["max_samples"] = 5000
config_file["training"]["batch_size"] = 8
config_file["training"]["num_workers"] = 8
total_batch_size = 24
n_trials = 12
upsample_factor = 4


def create_train_model(config_dict, train_loader, test_loader, device, acc_grads):
    config = config_dict

    model = FNO_2(
        in_channel=5,
        n_channels=config["n_channels"],
        n_residual_blocks=config["n_residual_blocks"],
        n_operator_blocks=config["n_operator_blocks"],
        modes=config["modes"],
        shifting_modes=config["shifting_modes"],
        apply_constraint=config["apply_constraint"],
        last_layer_constraint=config["last_layer_constraint"],
    )
    model = model.to(device)
    test_losses, train_losses, model = training_model(
        model,
        None,
        config_file["training"]["learning_rate"],
        config_file["training"]["epochs"],
        train_loader,
        test_loader=test_loader,
        use_amp=False,
        use_early_stopping=True,
        print_updates=False,
        gpu_id=device.index,
        accumulate_gradients=acc_grads,
        upsample_factor=upsample_factor,
    )
    return model


def create_dataset():
    dataset = dataset_sr(
        max_samples=config_file["training"]["max_samples"], use_normalizing=True
    )
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_file["training"]["batch_size"],
        shuffle=True,
        num_workers=config_file["training"]["num_workers"],
        persistent_workers=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config_file["training"]["batch_size"],
        shuffle=False,
        num_workers=config_file["training"]["num_workers"],
        persistent_workers=True,
        pin_memory=True,
    )

    print("data ready")
    return train_loader, test_loader, train_dataset, test_dataset


def objective_parent(
    train_loader, test_loader, train_dataset, test_dataset, **model_kwargs
):
    def objective(trial: trial.Trial):
        params = {
            "n_channels": trial.suggest_int("n_channels", 16, 64),
            "n_residual_blocks": trial.suggest_int("n_residual_blocks", 1, 6),
            "n_operator_blocks": trial.suggest_int("n_operator_blocks", 1, 10),
            "modes": trial.suggest_int("modes", 8, 32),
            "shifting_modes": trial.suggest_int("shifting_modes", 0, 10),
            "apply_constraint": trial.suggest_categorical(
                "apply_constraint", [True, False]
            ),
            "last_layer_constraint": trial.suggest_categorical(
                "last_layer_constraint", ["relu", "exp", "none"]
            ),
        }
        model = output = high_res = low_res = None
        initial_batch_size = config_file["training"]["batch_size"]
        current_batch_size = initial_batch_size
        train_loader_trial = train_loader  # use original loader by default
        test_loader_trial = test_loader

        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            model = output = high_res = low_res = None
            try:
                # only recreate loaders if batch size changed
                if current_batch_size != initial_batch_size:
                    train_loader_trial = DataLoader(
                        train_dataset,
                        batch_size=current_batch_size,
                        shuffle=True,
                        num_workers=config_file["training"]["num_workers"],
                        persistent_workers=True,
                        pin_memory=True,
                    )
                    test_loader_trial = DataLoader(
                        test_dataset,
                        batch_size=current_batch_size,
                        shuffle=False,
                        num_workers=config_file["training"]["num_workers"],
                        persistent_workers=True,
                        pin_memory=True,
                    )

                model = create_train_model(
                    params,
                    train_loader_trial,
                    test_loader_trial,
                    device,
                    total_batch_size // current_batch_size,
                )

                # evaluation
                model.eval()
                test_loss = 0
                for data in test_loader_trial:
                    high_res, low_res = data[0], data[1]
                    low_res = low_res.to(device, non_blocking=True)
                    high_res = high_res.to(device, non_blocking=True)
                    output = model(low_res, **model_kwargs)
                    test_loss += loss_fn(output, high_res).item()
                test_loss /= len(test_loader_trial)
                return test_loss
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        f"OOM encountered, reducing batch size from {current_batch_size} -> {max(1, current_batch_size // 2)}"
                    )
                    current_batch_size = max(1, current_batch_size // 2)
                    retry_count += 1
                    torch.cuda.empty_cache()
                else:
                    raise
            finally:
                for var in [model, output, high_res, low_res]:
                    if var is not None:
                        del var
                torch.cuda.empty_cache()

    return objective


if __name__ == "__main__":
    reserve_memory = torch.zeros([2, 4], dtype=torch.int32).to(device)
    print("preparing data")
    train_loader, test_loader, train_dataset, test_dataset = create_dataset()
    del reserve_memory
    loss_fn = MSELoss()
    experiment_folder = os.path.abspath("experiments/cfno_2/experiment_4/results")
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{os.path.join(experiment_folder, 'optuna_study.db')}",
        load_if_exists=True,
    )
    study.optimize(
        objective_parent(
            train_loader,
            test_loader,
            train_dataset,
            test_dataset,
            upsample_factor=upsample_factor,
        ),
        show_progress_bar=True,
        n_trials=n_trials,
    )
