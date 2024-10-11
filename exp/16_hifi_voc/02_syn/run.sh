#!/bin/bash -eu

bin_dir=../../../src/vocoders/bin

HYDRA_FULL_ERROR=1 python $bin_dir/synthesize_with_cf0_vuv.py \
    dataset=with_cf0_vuv \
    generator=hifi_vocoder \
    discriminator=bigvgan \
    lit_module=with_cf0_vuv \
    syn.ckpt_path=../01_train/out/ckpt/last.ckpt
