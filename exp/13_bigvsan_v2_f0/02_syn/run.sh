#!/bin/bash -eu

bin_dir=../../../src/vocoders/bin

HYDRA_FULL_ERROR=1 python $bin_dir/synthesize.py \
    generator=f0_aware_bigvgan_v2 \
    discriminator=bigvsan_v2 \
    ckpt_path=../../01_train/out/ckpt/last.ckpt