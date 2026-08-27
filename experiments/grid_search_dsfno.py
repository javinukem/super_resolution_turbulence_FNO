import gc
import json
import os
import sys
from itertools import product
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.training_cnn import training_model
from src.utils.pipeline import (
    build_dataset,
    build_train_test_loaders,
    instantiate_model,
)


@hydra.main(
    version_base="1.3",
    config_path="../configs/experiments",
    config_name="grid_search_dsfno",
)
def main(cfg: DictConfig) -> None:
    if cfg["runtime"]["autocvd"]["enabled"]:
        from autocvd import autocvd

        autocvd(num_gpus=int(cfg["runtime"]["autocvd"]["num_gpus"]))

    dataset = build_dataset(cfg["data"])
    train_loader, test_loader = build_train_test_loaders(
        dataset, cfg["data"], int(cfg["training"]["batch_size"])
    )

    keys = list(cfg["grid"].keys())
    grid = list(product(*cfg["grid"].values()))
    output_dir = cfg["output"]["root_dir"]
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.jsonl")

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        for values in grid:
            config_run = dict(zip(keys, values))
            model_params = OmegaConf.to_container(cfg["model"]["params"], resolve=True)
            model_params.update(config_run)
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
                use_float16=bool(cfg["training"]["use_float16"]),
                use_early_stopping=bool(cfg["training"]["use_early_stopping"]),
                early_stopping_patience=int(cfg["training"]["early_stopping_patience"]),
                early_stopping_delta=float(cfg["training"]["early_stopping_delta"]),
                upsample_factor=int(cfg["training"]["upsample_factor"]),
            )

            run_name = "_".join(f"{k}{v}" for k, v in config_run.items())
            run_dir = os.path.join(output_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)

            torch.save(model.state_dict(), os.path.join(run_dir, "weights.pt"))
            with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
                payload = {
                    "model": {
                        "module": cfg["model"]["module"],
                        "class_name": cfg["model"]["class_name"],
                        "params": model_params,
                    },
                    "training": OmegaConf.to_container(cfg["training"], resolve=True),
                    "grid_run": config_run,
                }
                import yaml

                yaml.safe_dump(payload, f, sort_keys=False)

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

            metric = {
                **config_run,
                "final_train_loss": train_losses[-1],
                "final_test_loss": test_losses[-1],
            }
            metrics_file.write(json.dumps(metric) + "\n")
            metrics_file.flush()

            del model, train_losses, test_losses
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
