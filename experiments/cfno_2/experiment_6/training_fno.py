import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.training.training_cnn import training_model
from src.utils.pipeline import (
    build_dataset,
    build_model,
    build_model_tag,
    build_train_test_loaders,
    create_run_dir,
    save_training_artifacts,
)


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/experiments",
    config_name="training_fno_exp6",
)
def main(cfg: DictConfig) -> None:
    if cfg["runtime"]["autocvd"]["enabled"]:
        from autocvd import autocvd

        autocvd(num_gpus=int(cfg["runtime"]["autocvd"]["num_gpus"]))

    dataset = build_dataset(cfg["data"])
    train_loader, test_loader = build_train_test_loaders(
        dataset, cfg["data"], int(cfg["training"]["batch_size"])
    )
    model = build_model(cfg["model"])

    test_losses, train_losses, best_model = training_model(
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
        upsample_factor=int(cfg["training"]["upsample_factor"]),
    )

    model_tag = build_model_tag(cfg["model"])
    run_dir, context = create_run_dir(cfg["runtime"], model_tag)
    payload = OmegaConf.to_container(cfg, resolve=True)
    save_training_artifacts(
        run_dir=run_dir,
        model=best_model,
        train_losses=train_losses,
        test_losses=test_losses,
        config_payload=payload,
        runtime_cfg=cfg["runtime"],
        context=context,
    )


if __name__ == "__main__":
    main()
