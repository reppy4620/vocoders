#!/bin/bash -eu

bin_dir=../../../src/vocoders/bin

HYDRA_FULL_ERROR=1 python $bin_dir/synthesize_with_f0.py \
    dataset=with_f0 \
    generator=f0_aware_bigvgan \
    discriminator=bigvsan_v2 \
    lit_module=san_with_f0 \
    loss=bigvsan_v2 \
    syn.ckpt_path=../01_train/out/ckpt/last.ckpt
