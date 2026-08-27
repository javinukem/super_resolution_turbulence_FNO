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
from experiments.cfno_2.experiment_2.fno_2_exp_2 import FNO_2
#from datetime import datetime
import gc

  # This brings 'src' into the path

def train_on_gpu(gpu_id, config_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = config_dict

    dataset = dataset_sr(max_samples=4000)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=config["training"]["batch_size"], persistent_workers=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"],
                             shuffle=False, num_workers=config["training"]["batch_size"], persistent_workers=False, pin_memory=True)

    model = FNO_2(
        in_channel=5,
        modes=config["fno_2"]["modes"],
        n_channels=config["fno_2"]["n_channels"],
        n_residual_blocks=config["fno_2"]["n_residual_blocks"],
        n_operator_blocks=config["fno_2"]["n_operator_blocks"],
        apply_constraint=config["fno_2"]["apply_constraint"],
        shifting_modes=config["fno_2"]["shifting_modes"],
        last_layer_kernel=config["fno_2"]["last_layer_kernel"],
        last_layer_constraint=config["fno_2"]["last_layer_constraint"]
    )
    model_vars = (
        f"m{config['fno_2']['modes']}_"
        f"nc{config['fno_2']['n_channels']}_"
        f"res{config['fno_2']['n_residual_blocks']}_"
        f"op{config['fno_2']['n_operator_blocks']}_"
        f"ac{int(config['fno_2']['apply_constraint'])}_"
        f"lk{config['fno_2']['last_layer_kernel']}"
        f"lc{config['fno_2']['last_layer_constraint']}"
    )

    test_losses, train_losses, model = training_model(
        model, None,
        config["training"]["learning_rate"],
        config["training"]["epochs"],
        train_loader, test_loader=test_loader,
        use_amp=False, upsample_factor=4, use_early_stopping=False
    )

    model_vars = (
        f"m{config['fno_2']['modes']}_"
        f"nc{config['fno_2']['n_channels']}_"
        f"res{config['fno_2']['n_residual_blocks']}_"
        f"op{config['fno_2']['n_operator_blocks']}_"
        f"ac{int(config['fno_2']['apply_constraint'])}_"
        f"lk{config['fno_2']['last_layer_kernel']}"
        f"lc{config['fno_2']['last_layer_constraint']}"
    )
    #date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_folder = os.path.join("experiments", "cfno_2", "experiment_2", f"{model_vars}")
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
