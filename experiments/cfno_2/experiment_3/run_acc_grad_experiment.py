from prepare_experiment import train_on_gpu
import yaml
import autocvd
import numpy as np
import os

autocvd.autocvd(1)

"""
Training the same model 4 times with different gradient accumulation to see the diference in loss evolution"""

acc_grads = [None, 3, 6, 9, 12]
train_losses = []
test_losses = []
times = []
base_config = yaml.safe_load(open("config.yaml", "r"))

for acc_grad in acc_grads:
    base_config["training"]["accumulate_gradients"] = acc_grad
    time, train_loss, test_loss = train_on_gpu(
        base_config, append_vars=f"{base_config['training']['accumulate_gradients']}"
    )
    times.append(time)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

os.makedirs(
    os.path.abspath("/export/data/jalegria/experiments/experiment_3/acc_grad"),
    exist_ok=True,
)
np.savez(
    os.path.abspath(
        "/export/data/jalegria/experiments/experiment_3/acc_grad/losses.npz"
    ),
    acc_grads=np.array(acc_grads),
    train_losses=np.array(train_losses),
    test_losses=np.array(test_losses),
    times=np.array(times),
)
