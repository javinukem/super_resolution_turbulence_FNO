import hydra
from omegaconf import DictConfig, OmegaConf

from src.training.training_cnn import training_model
from src.utils.pipeline import (
    build_dataset,
    build_model,
    build_model_tag,
    build_train_test_loaders,
    create_run_dir,
    save_training_artifacts,
)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    autocvd_cfg = cfg["runtime"]["autocvd"]
    if autocvd_cfg["enabled"]:
        from autocvd import autocvd

        autocvd(num_gpus=int(autocvd_cfg["num_gpus"]))

    dataset = build_dataset(cfg["data"])
    train_loader, test_loader = build_train_test_loaders(
        dataset, cfg["data"], int(cfg["training"]["batch_size"])
    )

    model = build_model(cfg["model"])

    train_kwargs = {
        "use_amp": bool(cfg["training"]["use_amp"]),
        "use_float16": bool(cfg["training"]["use_float16"]),
        "use_early_stopping": bool(cfg["training"]["use_early_stopping"]),
        "early_stopping_patience": int(cfg["training"]["early_stopping_patience"]),
        "early_stopping_delta": float(cfg["training"]["early_stopping_delta"]),
        "print_updates": bool(cfg["training"]["print_updates"]),
    }
    if cfg["training"]["accumulate_gradients"] is not None:
        train_kwargs["accumulate_gradients"] = int(cfg["training"]["accumulate_gradients"])
    if cfg["training"]["gpu_id"] is not None:
        train_kwargs["gpu_id"] = int(cfg["training"]["gpu_id"])
    if cfg["training"]["upsample_factor"] is not None:
        train_kwargs["upsample_factor"] = int(cfg["training"]["upsample_factor"])

    test_losses, train_losses, best_model = training_model(
        model=model,
        loss=None,
        learning_rate=float(cfg["training"]["learning_rate"]),
        epochs=int(cfg["training"]["epochs"]),
        train_loader=train_loader,
        test_loader=test_loader,
        **train_kwargs,
    )

    if cfg["runtime"]["output"]["enabled"]:
        model_tag = build_model_tag(cfg["model"])
        run_dir, context = create_run_dir(cfg["runtime"], model_tag)
        config_payload = OmegaConf.to_container(cfg, resolve=True)
        save_training_artifacts(
            run_dir=run_dir,
            model=best_model,
            train_losses=train_losses,
            test_losses=test_losses,
            config_payload=config_payload,
            runtime_cfg=cfg["runtime"],
            context=context,
        )


if __name__ == "__main__":
    main()
