import yaml
import json
import numpy as np
from copy import deepcopy

from gen_arithmetic_data import gen_noisy_dataset


if __name__ == "__main__":
    base_config_path = "configs/dataset_configs/default.yaml"
    with open(base_config_path, "r") as f:
      base_config = yaml.safe_load(f)

    print("Base config: ")
    print(json.dumps(base_config, indent=2))

    ops = {
            #"add": 10,  # max len of integer args for op
            "mul": 8,
            "div": 8,
            "median": 10,
          }

    d_noises = np.linspace(0, 1, num=3, endpoint=True)
    c_noises = np.linspace(0.25, 0.75, num=3, endpoint=True)
    l_noises = np.linspace(0.25, 0.75, num=3, endpoint=True)
    dynamic_noises = np.linspace(0.25, 0.75, num=3, endpoint=True)

    def update_and_generate(updates: dict):
       exp_config = deepcopy(base_config)
       exp_config.update(updates)
       gen_noisy_dataset(**exp_config)

    # Create prompting datasets
    exp_config = dict()
    for op, arg_max_size in ops.items():
        exp_config["op_name"] = op
        exp_config["arg_max_size"] = arg_max_size
        for d_noise in d_noises:
          exp_config["doc_noise"] = d_noise
          if d_noise == 0:
              update_and_generate(exp_config)
          else:
            for c_noise in c_noises:
               exp_config["char_noise"] = c_noise
               update_and_generate(exp_config)
            # Reset char noise
            exp_config["char_noise"] = base_config["char_noise"]
            for l_noise in l_noises:
               exp_config["line_noise"] = l_noise
               update_and_generate(exp_config)
            # Reset line noise
            exp_config["line_noise"] = base_config["line_noise"]
            for dynamic_noise in dynamic_noises:
               exp_config["dynamic_noise"] = dynamic_noise
               update_and_generate(exp_config)
            # Reset dynamic noise
            exp_config["dynamic_noise"] = base_config["dynamic_noise"]
        # Reset dataset nosie
        exp_config["doc_noise"] = base_config["doc_noise"]
            