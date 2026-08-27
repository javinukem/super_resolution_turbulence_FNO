# train_one_model.py
import os
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.dataloader.dataloader_3d import dataset_sr
from src.training.training_cnn import training_model
from experiments.edsr.experiment_1.models_edsr import EDSR

# from datetime import datetime
import gc

# This brings 'src' into the path


def train_on_gpu(gpu_id, config_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = config_dict

    dataset = dataset_sr(max_samples=4000)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["batch_size"],
        persistent_workers=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["batch_size"],
        persistent_workers=False,
        pin_memory=True,
    )

    model = EDSR(
        input_channels=5,
        n_resblocks=config["edsr"]["n_resblocks"],
        n_feats=config["edsr"]["n_feats"],
        kernel_size=config["edsr"]["kernel_size"],
        activation_f=config["edsr"]["activation_f"],
    )
    model_vars = (
        f"resb{config['edsr']['n_resblocks']}_"
        f"feats{config['edsr']['n_feats']}_"
        f"ks{config['edsr']['kernel_size']}_"
        f"af{config['edsr']['activation_f']}"
    )

    test_losses, train_losses, model = training_model(
        model,
        None,
        config["training"]["learning_rate"],
        config["training"]["epochs"],
        train_loader,
        test_loader=test_loader,
        use_amp=True,
        use_early_stopping=False,
        print_updates=True,
    )

    # date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_folder = os.path.join(
        "experiments", "edsr", "experiment_1", "models", f"{model_vars}"
    )
    os.makedirs(exp_folder, exist_ok=True)

    with open(os.path.join(exp_folder, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    torch.save(model.state_dict(), os.path.join(exp_folder, "weights.pt"))

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Test Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, "loss_curve.png"))
    plt.close()

    print(f"[GPU {gpu_id}] Finished training {model_vars}")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    del model
    gc.collect()
