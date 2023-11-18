#!/bin/bash

loss_file=../../01_train/out/logs/csv/version_1/metrics.csv
out_dir=../out

script_dir=../../../../vocoders

python $script_dir/bin/plot_loss.py \
    --loss_file $loss_file \
    --out_dir $out_dir
