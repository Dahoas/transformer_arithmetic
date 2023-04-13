#!/bin/bash

data_path=$1

#conda activate base

cd /mnt/nvme/home/alex/repos/maia/transformer_arithmetic

sleep 5

touch test_launch_dir/${data_path}.txt
hostname >> test_launch_dir/${data_path}.txt

