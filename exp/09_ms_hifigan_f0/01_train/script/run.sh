#!/bin/bash -eu

script_dir=../../../../vocoders

HYDRA_FULL_ERROR=1 python $script_dir/bin/train.py \
    dataset=with_f0 \
    generator=ms_f0_aware_hifigan \
    discriminator=bigvgan \
    lit_module=mb_with_f0
