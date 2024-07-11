#!/bin/bash -eu

bin_dir=../../../src/vocoders/bin

HYDRA_FULL_ERROR=1 python $bin_dir/synthesize_with_f0.py \
    generator=ms_f0_aware_bigvgan \
    discriminator=bigvgan \
    lit_module=mb_with_f0 \
    ckpt_path=../../01_train/out/ckpt/last.ckpt