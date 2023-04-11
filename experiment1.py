import os
import argparse
import yaml
import subprocess
from threading import Thread
from threading import Semaphore


def task(semaphore, command):
    with semaphore:
        proc = subprocess.Popen(command, shell=True)
        proc.wait()

if __name__ == "__main__":
    sem = Semaphore(2)

    experiment_config = "configs/experiment_configs/config.yaml"
    with open(experiment_config, "r") as f:
        experiment_config = yaml.safe_load(f)
        print(experiment_config)

    def make_experiment_list(experiment_config):
        #return ['sleep ' + str(3*i+5) for i in range(10)]
        return ['bash scripts/launch_test.sh ' + str(i) for i in range(10)]

    experiment_list = make_experiment_list(experiment_config)
    for experiment in experiment_list:
        worker = Thread(target=task, args=(sem, experiment))
        worker.start()


