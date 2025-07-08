#!/bin/bash -eu

bin_dir=../../../src/vocoders/bin

HYDRA_FULL_ERROR=1 uv run $bin_dir/train.py \
    dataset=with_f0 \
    generator=f0_aware_bigvgan_more \
    discriminator=bigvgan \
    lit_module=with_f0
