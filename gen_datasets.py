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
            #"add": {"arg_max_size": 10, "invisible_ops": [], "visible_ops": [], "n_args": 2},
            #"mul": {"arg_max_size": 5, "invisible_ops": ["__add__", "__sub__"], "visible_ops": [], "n_args": 2},
            #"div": {"arg_max_size": 8, "invisible_ops": ["__add__", "__sub__"], "visible_ops": [], "n_args": 2},
            "median": {"arg_max_size": 3, "invisible_ops": ["__add__", "__sub__", "__mul__", "__floordiv__"], "visible_ops": [], "n_args": 5},
          }

    d_noises = [0, 0.2, 0.5, 0.8]
    c_noises = [0.3, 0.6, 0.9]
    l_noises = [0.3, 0.6, 0.9]
    dynamic_noises = [0.2, 0.4]

    def update_and_generate(updates: dict):
       exp_config = deepcopy(base_config)
       exp_config.update(updates)
       gen_noisy_dataset(**exp_config)

    # Create prompting datasets
    exp_config = dict()
    for op in ops:
        exp_config["op_name"] = op
        exp_config["arg_max_size"] = ops[op]["arg_max_size"]
        exp_config["invisible_ops"] = ops[op]["invisible_ops"]
        exp_config["visible_ops"] = ops[op]["visible_ops"]
        exp_config["n_args"] = ops[op]["n_args"]
        for d_noise in d_noises:
          exp_config["doc_noise"] = d_noise
          if d_noise == 0:
            update_and_generate(exp_config)
            pass
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
        # Reset dataset noise
        exp_config["doc_noise"] = base_config["doc_noise"]
            