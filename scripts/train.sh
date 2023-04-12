#!/bin/bash

deepspeed --num_gpus=8 train.py --data_path data/datasets/additions_chain_of_thought_template_0.0_0.0_0.0 \
--sparsity_modes agglomerative agglomerative agglomerative agglomerative --sparsity_scheme 1 2 3 4 \
--epochs 5 \
--deepspeed configs/ds_config.json
