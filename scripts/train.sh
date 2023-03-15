deepspeed --num_gpus=8 train.py --data_path data/additions_0.0_simple_template.jsonl \
--deepspeed configs/ds_config.json