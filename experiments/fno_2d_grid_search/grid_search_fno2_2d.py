import gc
import json
import os
import sys
from itertools import product
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.training_cnn import training_model
from src.utils.pipeline import build_dataset, instantiate_model


def compute_loss(model, data_loader, criterion, upsample_factor):
    model.cpu().eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            high_res = data[0]
            low_res = data[1]
            pred = model(low_res, upsample_factor)
            total_loss += criterion(high_res, pred).item()
    return total_loss / len(data_loader.dataset)


@hydra.main(
    version_base="1.3",
    config_path="../../configs/experiments",
    config_name="grid_search_fno2_2d",
)
def main(cfg: DictConfig) -> None:
    if cfg["runtime"]["autocvd"]["enabled"]:
        from autocvd import autocvd

        autocvd(num_gpus=int(cfg["runtime"]["autocvd"]["num_gpus"]))

    dataset = build_dataset(cfg["data"])
    dataset_len = len(dataset)
    train_size = int(float(cfg["data"]["splits"]["train"]) * dataset_len)
    test_size = int(float(cfg["data"]["splits"]["test"]) * dataset_len)
    val_size = dataset_len - train_size - test_size
    train_dataset, test_dataset, val_dataset = random_split(
        dataset, [train_size, test_size, val_size]
    )

    loaders_cfg = cfg["data"]["loaders"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(loaders_cfg["train_num_workers"]),
        persistent_workers=bool(loaders_cfg["persistent_workers"]),
        pin_memory=bool(loaders_cfg["pin_memory"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(loaders_cfg["test_num_workers"]),
        persistent_workers=False,
        pin_memory=bool(loaders_cfg["pin_memory"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(loaders_cfg["val_num_workers"]),
        persistent_workers=False,
        pin_memory=bool(loaders_cfg["pin_memory"]),
    )

    param_grid = OmegaConf.to_container(cfg["grid"], resolve=True)
    keys = list(param_grid.keys())
    grid = list(product(*param_grid.values()))

    results = []
    output_root = cfg["output"]["root_dir"]
    os.makedirs(output_root, exist_ok=True)
    upsample_factor = int(cfg["training"]["upsample_factor"])

    for values in grid:
        config_run = dict(zip(keys, values))
        model_params = OmegaConf.to_container(cfg["model"]["params"], resolve=True)
        model_params.update(
            {
                "modes": config_run["modes"],
                "n_channels": config_run["chan"],
                "n_residual_blocks": config_run["res"],
                "n_operator_blocks": config_run["op"],
                "last_layer_constraint": config_run["last_layer_constraint"],
            }
        )
        model = instantiate_model(
            module=cfg["model"]["module"],
            class_name=cfg["model"]["class_name"],
            params=model_params,
        )

        test_losses, train_losses, model = training_model(
            model=model,
            loss=None,
            learning_rate=float(cfg["training"]["learning_rate"]),
            epochs=int(cfg["training"]["epochs"]),
            train_loader=train_loader,
            test_loader=test_loader,
            use_amp=bool(cfg["training"]["use_amp"]),
            use_early_stopping=bool(cfg["training"]["use_early_stopping"]),
            early_stopping_patience=int(cfg["training"]["early_stopping_patience"]),
            early_stopping_delta=float(cfg["training"]["early_stopping_delta"]),
            upsample_factor=upsample_factor,
        )

        val_loss = compute_loss(model, val_loader, torch.nn.MSELoss(), upsample_factor)
        run_name = "_".join(f"{k}{v}" for k, v in config_run.items())
        run_dir = os.path.join(output_root, run_name)
        os.makedirs(run_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(run_dir, "weights.pt"))
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "model": {
                        "module": cfg["model"]["module"],
                        "class_name": cfg["model"]["class_name"],
                        "params": model_params,
                    },
                    "training": OmegaConf.to_container(cfg["training"], resolve=True),
                    "grid_run": config_run,
                },
                f,
                sort_keys=False,
            )

        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses, label="Test Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Losses")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "loss_curve.png"))
        plt.close()

        result = {
            **config_run,
            "final_train_loss": train_losses[-1],
            "final_test_loss": test_losses[-1],
            "validation_loss": val_loss,
        }
        results.append(result)
        with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        del model, train_losses, test_losses
        torch.cuda.empty_cache()
        gc.collect()

    results_path = os.path.join(output_root, "results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    plt.figure(figsize=(10, 6))
    for key in ["res", "op"]:
        for mode in set(r["modes"] for r in results):
            x = [r[key] for r in results if r["modes"] == mode]
            y = [r["validation_loss"] for r in results if r["modes"] == mode]
            plt.plot(x, y, marker="o", label=f"{key.upper()} vs Loss (modes={mode})")
    plt.xlabel("Block Count")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Block Count for Different Modes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, "summary_plot.png"))
    plt.close()


if __name__ == "__main__":
    main()
