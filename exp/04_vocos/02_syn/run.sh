#!/bin/bash -eu

bin_dir=../../../src/vocoders/bin

HYDRA_FULL_ERROR=1 python $bin_dir/synthesize.py \
    generator=vocos \
    discriminator=bigvgan \
    ckpt_path=../../01_train/out/ckpt/last.ckpt
