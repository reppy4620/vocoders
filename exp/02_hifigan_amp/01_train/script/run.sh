#!/bin/bash -eu

script_dir=../../../../vocoders

HYDRA_FULL_ERROR=1 python $script_dir/bin/train.py \
    generator=bigvgan \
    discriminator=hifigan
