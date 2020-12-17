#!/usr/bin/env bash
model_name='attribute_attention'
for seed in 12 22 32 42 52
do
    for prompt in {1..8}
    do
        python train_CTS.py --test_prompt_id ${prompt} --model_name ${model_name} --seed ${seed}
    done
done