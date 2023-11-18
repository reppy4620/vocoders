#!/bin/bash -eu

script_dir=../../../../vocoders

HYDRA_FULL_ERROR=1 python $script_dir/bin/train.py \
    dataset=with_f0 \
    generator=f0_aware_bigvgan \
    discriminator=bigvgan \
    lit_module=with_f0
