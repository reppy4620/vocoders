#!/bin/bash -eu

bin_dir=../../../src/vocoders/bin

HYDRA_FULL_ERROR=1 uv run $bin_dir/train.py \
    dataset=with_f0 \
    generator=ms_istft_f0_aware_bigvgan \
    discriminator=bigvgan \
    lit_module=mb_with_f0
