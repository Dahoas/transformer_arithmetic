#!/bin/bash

deepspeed --num_gpus=8 train_sort.py --data_path data/datasets/sort_chain_of_thought_template_100000 \
--epochs 20 \
--deepspeed configs/ds_config.json
