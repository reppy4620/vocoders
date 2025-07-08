#!/bin/bash

loss_file=../../01_train/out/logs/csv/version_1/metrics.csv
out_dir=../out

bin_dir=../../../src/vocoders/bin

uv run $bin_dir/plot_loss.py \
    --loss_file $loss_file \
    --out_dir $out_dir
