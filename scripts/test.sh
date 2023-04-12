deepspeed --num_gpus=1 train.py --data_path data/datasets/additions_chain_of_thought_template_0.0_0.0_0.0 \
--metric_data_size 10  --train_data_size 100 \
--sparsity_modes agglomerative --sparsity_scheme 1 \
--deepspeed configs/ds_config.json
