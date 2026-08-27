# run_parallel_grid.py

# Set the number of GPUs you want to use
DESIRED_NUM_GPUS = 5  # Change this to the number of GPUs you want

from multiprocessing import Process, Lock, Manager, Queue
from itertools import product
import yaml
from train_one_model import train_on_gpu
import copy
import torch
import sys
import os
import time
import psutil
import subprocess
from datetime import datetime
from multiprocessing import Value

def get_model_vars(config):
    """Generate model variable string from config"""
    return (
        f"m{config['fno_2']['modes']}_"
        f"nc{config['fno_2']['n_channels']}_"
        f"res{config['fno_2']['n_residual_blocks']}_"
        f"op{config['fno_2']['n_operator_blocks']}_"
        f"ac{int(config['fno_2']['apply_constraint'])}_"
        f"lk{config['fno_2']['last_layer_kernel']}"
        f"lc{config['fno_2']['last_layer_constraint']}"
    )


def is_model_already_trained(config):
    """Check if a model with given config has already been trained"""
    model_vars = get_model_vars(config)
    exp_folder = os.path.join("experiments", "cfno_2", "experiment_2", f"{model_vars}")
    
    # Check if all required files exist
    required_files = ["config.yaml", "weights.pt", "loss_curve.png"]
    
    if not os.path.exists(exp_folder):
        return False
    
    for file in required_files:
        if not os.path.exists(os.path.join(exp_folder, file)):
            return False
    
    return True

def filter_completed_configs(configs):
    """Filter out configs that have already been completed"""
    pending_configs = []
    skipped_count = 0
    
    print("Checking for already completed models...")
    for config in configs:
        if is_model_already_trained(config):
            skipped_count += 1
        else:
            pending_configs.append(config)
    
    print(f"Found {skipped_count} already completed models. Skipping them.")
    print(f"Remaining configurations to train: {len(pending_configs)}")
    
    return pending_configs


def get_available_gpus(mem_threshold_mb=200):
    """
    Returns list of available GPU indices with allocated memory < threshold (in MB).
    Uses torch APIs that don't require NVML.
    """
    available = []
    if not torch.cuda.is_available():
        return available

    for i in range(torch.cuda.device_count()):
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**2  # Convert to MB
        print(i, mem_allocated)
        if mem_allocated < mem_threshold_mb:
            available.append(i)

    return available

# Configuration
MAX_PROCESSES_PER_GPU = 1
# Memory threshold (in MiB) to consider a GPU "free"
GPU_MEMORY_THRESHOLD = 50
VERBOSE = False

# Load and prepare configs
base_config = yaml.safe_load(open("config.yaml", "r"))
grid_config = base_config["grid_fno2"]

# Build parameter grid
keys = list(grid_config.keys())
values = list(grid_config.values())
grid = list(product(*values))

# Construct full configs
all_configs = []
for combo in grid:
    config = copy.deepcopy(base_config)
    for k, v in zip(keys, combo):
        if k == "chan":
            config["fno_2"]["n_channels"] = v
        elif k == "res":
            config["fno_2"]["n_residual_blocks"] = v
        elif k == "op":
            config["fno_2"]["n_operator_blocks"] = v
        elif k == "modes":
            config["fno_2"]["modes"] = v
        elif k == "last_layer_constraint":
            config["fno_2"]["last_layer_constraint"] = v
    all_configs.append(config)

# Filter out already completed models
configs = filter_completed_configs(all_configs)

if not configs:
    print("All models have already been trained! Nothing to do.")
    sys.exit(0)
    
if __name__ == "__main__":
    free = get_available_gpus(mem_threshold_mb=1000)
    print("Available GPUs:", free)