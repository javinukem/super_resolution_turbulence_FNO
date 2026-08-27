from __future__ import annotations

import importlib
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split


def _to_plain_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def instantiate_model(
    module: str, class_name: str, params: DictConfig | Mapping[str, Any] | None = None
) -> torch.nn.Module:
    model_module = importlib.import_module(module)
    model_class = getattr(model_module, class_name)
    return model_class(**_to_plain_dict(params))


def build_model(model_cfg: DictConfig | Mapping[str, Any]) -> torch.nn.Module:
    cfg = _to_plain_dict(model_cfg)
    return instantiate_model(
        module=cfg["module"],
        class_name=cfg["class_name"],
        params=cfg.get("params", {}),
    )


def build_dataset(data_cfg: DictConfig | Mapping[str, Any]):
    cfg = _to_plain_dict(data_cfg)
    kind = cfg["kind"]
    kwargs = cfg.get("kwargs", {})

    if kind == "3d":
        from src.dataloader.dataloader_3d import dataset_sr
    elif kind == "2d":
        from src.dataloader.dataloader_2d_v2 import dataset_sr
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")

    return dataset_sr(**kwargs)


def build_train_test_loaders(dataset, data_cfg: DictConfig, batch_size: int):
    train_split = float(data_cfg["train_split"])
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    loaders_cfg = data_cfg["loaders"]
    train_workers = int(loaders_cfg["train_num_workers"])
    test_workers = int(loaders_cfg["test_num_workers"])
    pin_memory = bool(loaders_cfg.get("pin_memory", True))
    persistent_workers = bool(loaders_cfg.get("persistent_workers", True))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        persistent_workers=persistent_workers and train_workers > 0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=test_workers,
        persistent_workers=persistent_workers and test_workers > 0,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def _format_context(params: Mapping[str, Any]) -> dict[str, Any]:
    context = dict(params)
    for key, value in params.items():
        if isinstance(value, bool):
            context[f"{key}_int"] = int(value)
    return context


def build_model_tag(model_cfg: DictConfig | Mapping[str, Any]) -> str:
    cfg = _to_plain_dict(model_cfg)
    params = _to_plain_dict(cfg.get("params", {}))
    template = cfg.get("tag_template")
    if not template:
        return str(cfg.get("name", "model"))
    return template.format(**_format_context(params))


def create_run_dir(runtime_cfg: DictConfig, model_tag: str) -> tuple[Path, dict[str, str]]:
    timestamp = datetime.now().strftime(runtime_cfg["timestamp_format"])
    date = datetime.now().strftime(runtime_cfg["date_format"])
    context = {"timestamp": timestamp, "date": date, "model_tag": model_tag}

    output_cfg = runtime_cfg["output"]
    root = Path(output_cfg["root_dir"])
    run_dir = root / output_cfg["run_dir_template"].format(**context)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, context


def save_training_artifacts(
    run_dir: Path,
    model: torch.nn.Module,
    train_losses: list[float],
    test_losses: list[float] | None,
    config_payload: Mapping[str, Any],
    runtime_cfg: DictConfig,
    context: Mapping[str, str],
) -> None:
    output_cfg = runtime_cfg["output"]

    if bool(output_cfg.get("save_config", True)):
        config_name = output_cfg.get("config_name", "config.yaml")
        with open(run_dir / config_name, "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(config_payload), f, sort_keys=False)

    if bool(output_cfg.get("save_weights", True)):
        weights_name = output_cfg.get("weights_name_template", "weights_{date}.pt").format(
            **context
        )
        torch.save(model.state_dict(), run_dir / weights_name)

    if bool(output_cfg.get("save_loss_curve", True)):
        import matplotlib.pyplot as plt

        loss_curve_name = output_cfg.get("loss_curve_name_template", "loss_curve.png").format(
            **context
        )
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        if test_losses is not None:
            plt.plot(test_losses, label="Test Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train vs Test Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / loss_curve_name)
        plt.close()


def load_model_from_folder(
    model_folder: str | Path,
    device: torch.device,
    legacy_module: str = "experiments.cfno_2.experiment_4.fno_2_exp_2",
    legacy_class_name: str = "FNO_2",
    weights_name: str = "weights.pt",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    folder = Path(model_folder)
    config_path = folder / "config.yaml"
    weights_path = folder / weights_name

    with open(config_path, "r", encoding="utf-8") as f:
        saved_cfg = yaml.safe_load(f)

    if "model" in saved_cfg and {"module", "class_name"} <= set(saved_cfg["model"].keys()):
        model_cfg = saved_cfg["model"]
    elif "fno_2" in saved_cfg:
        model_cfg = {
            "module": legacy_module,
            "class_name": legacy_class_name,
            "params": saved_cfg["fno_2"],
        }
    else:
        raise ValueError(f"Unsupported saved model config format in {config_path}")

    model = instantiate_model(
        module=model_cfg["module"],
        class_name=model_cfg["class_name"],
        params=model_cfg.get("params", {}),
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model, saved_cfg
