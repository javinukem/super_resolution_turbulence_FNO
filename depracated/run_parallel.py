# run_parallel_grid.py

from autocvd import autocvd
autocvd(num_gpus=2)
from multiprocessing import Process
from itertools import product
import yaml
from train_one_model import train_on_gpu
import copy
import torch
import sys
import os


num_gpus = torch.cuda.device_count()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
base_config = yaml.safe_load(open("config.yaml", "r"))

grid_config = base_config["grid_fno2"]

# Build parameter grid
keys = list(grid_config.keys())
values = list(grid_config.values())
grid = list(product(*values))

# Construct full configs
configs = []
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
    configs.append(config)

# Launch training in parallel batches
if __name__ == "__main__":
    processes = []
    for i, config in enumerate(configs):
        gpu_id = i % num_gpus
        p = Process(target=train_on_gpu, args=(gpu_id, config))
        p.start()
        processes.append(p)

        # Optional: wait for full batch to finish before next one
        if (i + 1) % num_gpus == 0:
            for p in processes:
                p.join()
            processes = []

    # Final join for leftover processes
    for p in processes:
        p.join()
