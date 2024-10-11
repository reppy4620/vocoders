#!/bin/bash -eu

bin_dir=../../../src/vocoders/bin

HYDRA_FULL_ERROR=1 python $bin_dir/train.py \
    dataset=with_cf0_vuv \
    generator=hifi_vocoder \
    discriminator=bigvgan \
    lit_module=with_cf0_vuv
