#!/bin/bash

deepspeed --num_gpus=8 train.py --data_path data/datasets/mul_10_10_chain_of_thought_template_100000 \
--epochs 1 --train_data_size 10000 --metric_batch_size 1 \
--deepspeed configs/ds_config.json \
--gradient_checkpointing 1
#--model_path ckpts/add_10_10_chain_of_thought_template_100000_10000_1_model
