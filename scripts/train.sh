#!/bin/bash

deepspeed --num_gpus=8 train.py --data_path data/datasets/additions_chain_of_thought_template_0.275_0.275_0.8 \
--deepspeed configs/ds_config.json
