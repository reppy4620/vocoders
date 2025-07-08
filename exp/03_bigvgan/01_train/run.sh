#!/bin/bash -eu

bin_dir=../../../src/vocoders/bin

HYDRA_FULL_ERROR=1 uv run $bin_dir/train.py \
    generator=bigvgan \
    discriminator=bigvgan
