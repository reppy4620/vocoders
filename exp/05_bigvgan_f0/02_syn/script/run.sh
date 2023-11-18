#!/bin/bash -eu

script_dir=../../../../vocoders

HYDRA_FULL_ERROR=1 python $script_dir/bin/synthesize_with_f0.py \
    generator=f0_aware_bigvgan \
    discriminator=bigvgan \
    lit_module=with_f0 \
    ckpt_path=../../01_train/out/ckpt/last.ckpt
