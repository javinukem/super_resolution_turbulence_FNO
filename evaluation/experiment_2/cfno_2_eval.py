import os
import sys
from pathlib import Path

import hydra

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
import time
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.pipeline import build_dataset, load_model_from_folder


def create_val_loader(val_dataset, batch_size, num_workers):
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )


@hydra.main(
    version_base="1.3", config_path="../../configs/evaluation", config_name="cfno_2_eval"
)
def main(cfg: DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["runtime"]["cuda_visible_devices"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_folder = cfg["paths"]["experiment_folder"]
    subfolders = [f.path for f in os.scandir(experiment_folder) if f.is_dir() and f.path]

    loss = torch.nn.MSELoss()
    dataset = build_dataset(cfg["data"])
    val_start = int(cfg["data"]["validation_start_index"])
    val_dataset = Subset(dataset, range(val_start, len(dataset)))

    initial_batch_size = int(cfg["evaluation"]["initial_batch_size"])
    min_batch_size = int(cfg["evaluation"]["min_batch_size"])
    num_workers = int(cfg["evaluation"]["num_workers"])

    print(f"Validation dataset size: {len(val_dataset)}")
    results = []

    for model_folder in subfolders:
        model_name = os.path.basename(model_folder)
        config_path = os.path.join(model_folder, "config.yaml")
        weights_path = os.path.join(model_folder, "weights.pt")
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            print(f"Skipping {model_folder}: missing config or weights")
            continue

        model, model_config = load_model_from_folder(model_folder=model_folder, device=device)
        model_cfg = model_config.get("model", {})
        model_params = model_cfg.get("params", model_config.get("fno_2", {}))

        current_batch_size = initial_batch_size
        while current_batch_size >= min_batch_size:
            try:
                print(f"Evaluating {model_name} with batch size {current_batch_size}")
                val_loader = create_val_loader(val_dataset, current_batch_size, num_workers)

                total_loss = 0.0
                i = 0
                start = time.time()
                with torch.no_grad():
                    for data in val_loader:
                        high_res = data[0].to(device)
                        low_res = data[1].to(device)
                        pred = model(low_res, int(cfg["evaluation"]["upsample_factor"]))
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

                results.append(
                    {
                        "folder": os.path.basename(model_folder),
                        "val_loss": val_loss,
                        "time": time_per_item,
                        "modes": model_params.get("modes"),
                        "n_channels": model_params.get("n_channels"),
                        "n_residual_blocks": model_params.get("n_residual_blocks"),
                        "n_operator_blocks": model_params.get("n_operator_blocks"),
                        "apply_constraint": model_params.get("apply_constraint"),
                        "shifting_modes": model_params.get("shifting_modes"),
                        "last_layer_kernel": model_params.get("last_layer_kernel"),
                        "last_layer_constraint": model_params.get(
                            "last_layer_constraint"
                        ),
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
                raise

    df = pd.DataFrame(results)
    output_file = os.path.join(experiment_folder, "fno_eval_results.csv")
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
