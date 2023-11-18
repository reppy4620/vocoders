#!/bin/bash -eu

script_dir=../../../../vocoders

HYDRA_FULL_ERROR=1 python $script_dir/bin/train.py \
    dataset=with_f0 \
    generator=ms_f0_aware_hifigan \
    discriminator=bigvsan \
    lit_module=san_mb_with_f0
