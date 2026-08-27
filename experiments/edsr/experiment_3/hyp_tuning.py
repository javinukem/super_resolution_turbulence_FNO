# train_one_model.py
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
from experiments.edsr.experiment_3.models_edsr import EDSR

# from datetime import datetime
import gc
import optuna
from optuna import trial
import multiprocessing as mp
# This brings 'src' into the path

config_file = yaml.safe_load(open("config.yaml", "r"))


def create_train_model(config_dict, train_loader, test_loader, device, acc_grads):
    config = config_dict

    model = EDSR(
        input_channels=5,
        n_resblocks=config["n_resblocks"],
        n_feats=config["n_feats"],
        kernel_size=config["kernel_size"],
        activation_f=config["activation_f"],
    )
    model = model.to(device)
    test_losses, train_losses, model = training_model(
        model,
        None,
        config_file["training"]["learning_rate"],
        config_file["training"]["epochs"],
        train_loader,
        test_loader=test_loader,
        use_amp=True,
        use_early_stopping=True,
        print_updates=True,
        gpu_id=device.index,
        accumulate_gradients=acc_grads,
    )
    return model


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
loss_fn = MSELoss()


def objective(trial):
    params = {
        "n_resblocks": trial.suggest_int("n_resblocks", 1, 8),
        "n_feats": trial.suggest_int("feats", 32, 128),
        "kernel_size": trial.suggest_int("kernel_size", 3, 5, step=2),
        "activation_f": trial.suggest_categorical(
            "activation_f", [None, "relu", "prelu"]
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
                16 // current_batch_size,
            )

            # evaluation
            model.eval()
            test_loss = 0
            for data in test_loader_trial:
                high_res, low_res = data[0], data[1]
                low_res = low_res.to(device, non_blocking=True)
                high_res = high_res.to(device, non_blocking=True)
                output = model(low_res)
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


study = optuna.create_study(
    study_name="convolutional turbulence sr", direction="minimize"
)
study.optimize(objective, n_trials=20, show_progress_bar=True, n_jobs=1)

print("Best params: ", study.best_params)
print("Best value: ", study.best_value)
print("Best Trial: ", study.best_trial)
print("Trials: ", study.trials)


df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
df.to_csv("experiments/edsr/experiment_3/hyp_tuning.csv", index=False)
