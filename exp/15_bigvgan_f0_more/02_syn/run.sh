#!/bin/bash -eu

bin_dir=../../../src/vocoders/bin

HYDRA_FULL_ERROR=1 python $bin_dir/synthesize_with_f0.py \
    dataset=with_f0 \
    generator=f0_aware_bigvgan_more \
    discriminator=bigvgan \
    lit_module=with_f0 \
    syn.ckpt_path=../01_train/out/ckpt/last.ckpt
