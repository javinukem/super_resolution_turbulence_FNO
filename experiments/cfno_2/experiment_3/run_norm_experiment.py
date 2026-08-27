from prepare_experiment import train_on_gpu
import yaml
import autocvd

autocvd.autocvd(1)

base_config = yaml.safe_load(open("config.yaml", "r"))

time_wo_norm = train_on_gpu(base_config, use_normalizing=False)
time_w_norm = train_on_gpu(base_config, use_normalizing=True)

print(time_w_norm, time_wo_norm)
