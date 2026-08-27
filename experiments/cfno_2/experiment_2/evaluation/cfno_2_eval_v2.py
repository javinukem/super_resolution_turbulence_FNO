import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)
import yaml
from autocvd import autocvd

autocvd(num_gpus=1)
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
from torch.utils.data import DataLoader, Subset
from experiments.cfno_2.experiment_2.fno_2_exp_2 import FNO_2
from src.dataloader.dataloader_3d import dataset_sr
import time
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_folder = os.path.abspath("/export/home/jalegria/Thesis/turbulence_sr/experiments/cfno_2/experiment_2/evaluation") #directory of the experiment
models_folder = os.path.abspath("/export/data/jalegria/models/cfno_2")
subfolders = [f.path for f in os.scandir(models_folder) if f.is_dir() and f.path]

loss = torch.nn.MSELoss()
dataset = dataset_sr()
val_dataset = Subset(dataset, range(5000, len(dataset)))

initial_batch_size = 16
min_batch_size = 1


def create_val_loader(batch_size):
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )


print(f"Validation dataset size: {len(val_dataset)}")

# Load existing results if the file exists
results_csv_path = os.path.join(results_folder, "fno_eval_results.csv")
if os.path.exists(results_csv_path):
    df_existing = pd.read_csv(results_csv_path)
    evaluated_models = set(df_existing["folder"].tolist())
    print(f"Found {len(evaluated_models)} previously evaluated models.")
else:
    df_existing = pd.DataFrame()
    evaluated_models = set()
    print("No previously evaluated models found.")

# Keep appending new results to this list
new_results = []

for model_folder in subfolders:
    model_name = os.path.basename(model_folder)

    if model_name in evaluated_models:
        print(f"Skipping {model_name}: already evaluated")
        continue

    model_config_path = os.path.join(model_folder, "config.yaml")
    weights_path = os.path.join(model_folder, "weights.pt")

    if not os.path.exists(model_config_path) or not os.path.exists(weights_path):
        print(f"Skipping {model_folder}: missing config or weights")
        continue

    model_config = yaml.safe_load(open(model_config_path))
    fno_cfg = model_config["fno_2"]

    model = FNO_2(
        in_channel=5,
        modes=fno_cfg["modes"],
        n_channels=fno_cfg["n_channels"],
        n_residual_blocks=fno_cfg["n_residual_blocks"],
        n_operator_blocks=fno_cfg["n_operator_blocks"],
        apply_constraint=fno_cfg["apply_constraint"],
        shifting_modes=fno_cfg["shifting_modes"],
        last_layer_kernel=fno_cfg["last_layer_kernel"],
        last_layer_constraint=fno_cfg["last_layer_constraint"],
    ).to(device)

    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    current_batch_size = initial_batch_size
    while current_batch_size >= min_batch_size:
        try:
            print(f"Evaluating {model_name} with batch size {current_batch_size}")
            val_loader = create_val_loader(current_batch_size)

            total_loss = 0.0
            i = 0
            start = time.time()
            with torch.no_grad():
                for data in val_loader:
                    high_res = data[0].to(device)
                    low_res = data[1].to(device)
                    pred = model(low_res, 4)
                    total_loss += loss(high_res, pred).item()
                    del high_res, low_res, pred
                    i += 1
                    if i % 100 == 0:
                        print(
                            f"{model_name} {i} / {len(val_loader.dataset) // current_batch_size}",
                            end="\r",
                        )
            time_per_item = (time.time() - start) / len(val_loader.dataset)
            val_loss = total_loss / len(val_loader.dataset)

            print(f"Validation loss: {val_loss:.6f}")

            new_results.append(
                {
                    "folder": model_name,
                    "val_loss": val_loss,
                    "time": time_per_item,
                    "modes": fno_cfg["modes"],
                    "n_channels": fno_cfg["n_channels"],
                    "n_residual_blocks": fno_cfg["n_residual_blocks"],
                    "n_operator_blocks": fno_cfg["n_operator_blocks"],
                    "apply_constraint": fno_cfg["apply_constraint"],
                    "shifting_modes": fno_cfg["shifting_modes"],
                    "last_layer_kernel": fno_cfg["last_layer_kernel"],
                    "last_layer_constraint": fno_cfg["last_layer_constraint"],
                }
            )
            break

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(
                    f"OOM at batch size {current_batch_size}, trying smaller batch size..."
                )
                current_batch_size = current_batch_size // 2
                torch.cuda.empty_cache()
                continue
            else:
                raise

# Append new results to existing ones and save
df_new = pd.DataFrame(new_results)
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
df_combined.to_csv(results_csv_path, index=False)
print(f"Results saved to {results_csv_path}")
