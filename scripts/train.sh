#!/bin/bash

deepspeed --num_gpus=8 train.py --data_path data/datasets/mix_10_20_chain_of_thought_template_100000 \
--epochs 1 --train_data_size 1000 --metric_batch_size 1 \
--deepspeed configs/ds_config.json
