#!/bin/bash -eu

bin_dir=../../../src/vocoders/bin

HYDRA_FULL_ERROR=1 python $bin_dir/synthesize.py \
    generator=vocos \
    discriminator=bigvgan \
    scheduler=cosine_warmup \
    train.loss_coef.second_disc=0.1 \
    optimizer=adamw_vocos \
    lit_module=vocos \
    loss=vocos \
    syn.ckpt_path=../01_train/out/ckpt/last.ckpt
