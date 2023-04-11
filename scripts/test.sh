deepspeed --num_gpus=1 train.py --data_path data/datasets/additions_chain_of_thought_template_0.0_0.0_0.0 --train_data_size 100 \
--deepspeed configs/ds_config.json
