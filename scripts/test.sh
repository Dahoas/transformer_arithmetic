#!/bin/bash
deepspeed --num_gpus=1 train.py --data_path data/datasets/rsh_4_7_chain_of_thought_template_100000  \
--metric_data_size 10  --train_data_size 100 --metric_batch_size 2 \
--deepspeed configs/ds_config.json
