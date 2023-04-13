import os
import argparse
import yaml
import subprocess


if __name__ == "__main__":
    max_concurrent_experiments = 2
    experiment_config = "configs/experiment_configs/config.yaml"
    with open(experiment_config, "r") as f:
        experiment_config = yaml.safe_load(f)
        print(experiment_config)

    """def make_experiment_list(experiment_config):
        #return ['sleep ' + str(3*i+5) for i in range(10)]
        return ['bash scripts/launch_test.sh ' + str(i) for i in range(10)]

    experiment_list = make_experiment_list(experiment_config)
    experiment_chunks = 5
    print(experiment_list)
    for j in range(experiment_chunks):
        left_ind = j*max_concurrent_experiments
        right_ind = min((j+1)*max_concurrent_experiments, len(experiment_list))
        print(left_ind, right_ind)
        procs = [subprocess.Popen(experiment, shell=True) for experiment in experiment_list[left_ind: right_ind]]
        for p in procs:
            p.wait()"""

