#!/bin/bash -eu

bin_dir=../../../src/vocoders/bin

HYDRA_FULL_ERROR=1 python $bin_dir/train.py \
    dataset=with_f0 \
    generator=ms_f0_aware_hifigan \
    discriminator=bigvsan \
    lit_module=san_mb_with_f0 \
    loss=san
