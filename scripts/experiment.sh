#!/bin/bash

config_file=$1

conda activate base

cd /mnt/nvme/home/alex/repos/maia/transformer_arithmetic

deepspeed --num_gpus=8 train.py --config $config_file \
--deepspeed configs/ds_config.json
