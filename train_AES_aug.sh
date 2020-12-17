#!/usr/bin/env bash
for seed in 12 22 32 42 52
do
    for prompt in {1..8}
    do
        python train_AES_aug.py --test_prompt_id ${prompt} --seed ${seed}
    done
done