#!/bin/bash

deepspeed --num_gpus=8 train.py --data_path data/datasets/no_template_add_len_10_len_10 \
--epochs 50 --train_data_size 20000 --metric_batch_size 1 --metric_data_size 144 \
--deepspeed configs/ds_config.json \
--gradient_checkpointing 1

deepspeed --num_gpus=8 train.py --data_path data/datasets/no_template_sub_len_10_len_10 \
--epochs 50 --train_data_size 20000 --metric_batch_size 1 --metric_data_size 144 \
--deepspeed configs/ds_config.json \
--gradient_checkpointing 1

deepspeed --num_gpus=8 train.py --data_path data/datasets/no_template_mul_len_5_len_5 \
--epochs 50 --train_data_size 20000 --metric_batch_size 1 --metric_data_size 144 \
--deepspeed configs/ds_config.json \
--gradient_checkpointing 1

deepspeed --num_gpus=8 train.py --data_path data/datasets/no_template_div_len_5_len_5 \
--epochs 50 --train_data_size 20000 --metric_batch_size 1 --metric_data_size 144 \
--deepspeed configs/ds_config.json \
--gradient_checkpointing 1

deepspeed --num_gpus=8 train.py --data_path data/datasets/no_template_add_len_10_len_10_sub_len_10_len_10_mul_len_5_len_5_div_len_5_len_5 \
--epochs 50 --train_data_size 20000 --metric_batch_size 1 --metric_data_size 144 \
--deepspeed configs/ds_config.json \
--gradient_checkpointing 1

