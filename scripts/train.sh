#!/bin/bash

deepspeed --num_gpus=8 train.py --data_path data/datasets/additions_chain_of_thought_template_0.0_0.0_0.0 \
--sparsity_modes sequential sequential sequential sequential sequential sequential sequential --sparsity_scheme 1 2 3 4 5 6 7 \
--epochs 8 \
--deepspeed configs/ds_config.json
