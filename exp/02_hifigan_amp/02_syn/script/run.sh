#!/bin/bash -eu

script_dir=../../../../vocoders

HYDRA_FULL_ERROR=1 python $script_dir/bin/synthesize.py \
    generator=bigvgan \
    discriminator=hifigan \
    ckpt_path=../../01_train/out/ckpt/last.ckpt
