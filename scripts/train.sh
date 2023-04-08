#!/bin/bash

deepspeed --num_gpus=8 train.py --data_path data/datasets/additions_0.005_simple_template \
--deepspeed configs/ds_config.json
