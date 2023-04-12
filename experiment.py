import os
import argparse
import yaml
import subprocess
from threading import Thread
from threading import Semaphore
from itertools import product
import time


def task(semaphore, command):
    with semaphore:
        proc = subprocess.Popen(command, shell=True)
        proc.wait()

if __name__ == "__main__":
    sem = Semaphore(3)

    master_data_path = "data/datasets/"
    master_config_path = "configs/experiment_configs/"
    master_config = yaml.safe_load(open(os.path.join(master_config_path, "master/experiments.yaml"), "r"))

    def parse_experiments_config(master_config):
        exp_config_paths = []
        # First iterate through experiment groups
        cnt = 0

        for exp_group in master_config.values():
            keys = [key for key in exp_group]
            experiment_params = tuple([exp_group[key] for key in keys])
            experiment_tuples = list(product(*experiment_params))

            for experiment in experiment_tuples:
                param_dict = {keys[i][:-1]:experiment[i] for i in range(len(keys))}
                data_file_name = "{}_{}_{}_{}_{}".format(param_dict["task"], 
                                                         param_dict["prompt_template"],
                                                         param_dict["dnl"],
                                                         param_dict["snl"],
                                                         param_dict["enl"])
                data_path = os.path.join(master_data_path, data_file_name)
                exp_config = {"data_path": data_path,
                              "model_path": param_dict["model"],
                              "tok_path": param_dict["tok"],
                              "epochs": int(param_dict["epoch"])}
                exp_config_path = os.path.join(master_config_path, f"{cnt}.yaml")
                with open(exp_config_path, "w") as f:
                    yaml.dump(exp_config, f)
                exp_config_paths.append(exp_config_path)
                cnt += 1
            
        return exp_config_paths

    exp_config_paths = parse_experiments_config(master_config)
    EXP_LIMIT = 1
    exp_config_paths = exp_config_paths[:EXP_LIMIT]

    for exp_config_path in exp_config_paths:
        command = f"bash scripts/experiment_launcher.sh {exp_config_path}"
        worker = Thread(target=task, args=(sem, command))
        worker.start()
        time.sleep(5)
