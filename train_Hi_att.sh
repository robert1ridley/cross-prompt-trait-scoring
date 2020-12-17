#!/usr/bin/env bash
for seed in 12 22 32 42 52
do
    for prompt in {1..2}
    do
        for attribute in score content organization word_choice sentence_fluency conventions
        do
            python train_Hi_att.py --test_prompt_id ${prompt} --attribute_name ${attribute} --seed ${seed}
        done
    done

    for prompt in {3..6}
    do
        for attribute in score content prompt_adherence language narrativity
        do
            python train_Hi_att.py --test_prompt_id ${prompt} --attribute_name ${attribute} --seed ${seed}
        done
    done

    for attribute in score content organization conventions
    do
        python train_Hi_att.py --test_prompt_id 7 --attribute_name ${attribute} --seed ${seed}
    done

    for attribute in score content organization word_choice sentence_fluency conventions
    do
        python train_Hi_att.py --test_prompt_id 8 --attribute_name ${attribute} --seed ${seed}
    done
done