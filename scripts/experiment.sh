#!/bin/bash

data_path=$1

conda activate base

cd /mnt/nvme/home/alex/repos/maia/transformer_arithmetic

deepspeed --num_gpus=8 train.py --data_path $data_path \
--deepspeed configs/ds_config.json
