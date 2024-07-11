#!/bin/bash -eu

bin_dir=../../../src/vocoders/bin

HYDRA_FULL_ERROR=1 python $bin_dir/train.py \
    dataset=with_f0 \
    generator=f0_aware_bigvgan_v2 \
    discriminator=bigvgan_v2 \
    lit_module=with_f0 \
    loss=bigvgan_v2
