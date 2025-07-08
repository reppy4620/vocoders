#!/bin/bash -eu

bin_dir=../../../src/vocoders/bin

HYDRA_FULL_ERROR=1 uv run $bin_dir/synthesize_with_f0.py \
    dataset=with_f0 \
    generator=ms_f0_aware_hifigan \
    discriminator=bigvsan \
    lit_module=san_mb_with_f0 \
    loss=san \
    syn.ckpt_path=../01_train/out/ckpt/last.ckpt
