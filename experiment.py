import os
import argparse
import yaml
import subprocess
from threading import Thread
from threading import Semaphore
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
            for model in exp_group.get("models"):
                for tok in exp_group.get("toks"):
                    for task in exp_group.get("tasks"):
                        for prompt_template in exp_group.get("prompt_templates"):
                            for dnl in exp_group.get("dnls"):
                                for enl in exp_group.get("enls"):
                                    for snl in exp_group.get("snls"):
                                        for epoch in exp_group.get("epochs"):
                                            data_file_name = f"{task}_{prompt_template}_{dnl}_{snl}_{enl}"
                                            data_path = os.path.join(master_data_path, data_file_name)
                                            exp_config = {
                                                           "data_path": data_path, 
                                                           "model_path": model,
                                                           "tok_path": tok,
                                                           "epochs": int(epoch),
                                                         }
                                            exp_config_path = os.path.join(master_config_path, f"{cnt}.yaml")
                                            with open(exp_config_path, "w") as f:
                                                yaml.dump(exp_config, f)
                                            exp_config_paths.append(exp_config_path)
                                            cnt += 1
        
        return exp_config_paths

    exp_config_paths = parse_experiments_config(master_config)
    EXP_LIMIT = 300
    exp_config_paths = exp_config_paths[:EXP_LIMIT]

    for exp_config_path in exp_config_paths:
        command = f"bash scripts/experiment_launcher.sh {exp_config_path}"
        worker = Thread(target=task, args=(sem, command))
        worker.start()
        time.sleep(5)
