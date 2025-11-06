#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=1


# 0.1  0.3 0.4 0.5 0.6 0.7 0.8 0.9

ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

model=meta-llama/Llama-2-7b-hf
# model="Enoch/llama-7b-hf"
model_name=$(echo "$model" | awk -F'/' '{print $2}')

run_asvd()
{
    python asvd.py \
    --model_id ${model} \
    --act_aware --alpha 0.5 \
    --n_calib_samples 256 \
    --scaling_method fisher \
    --eval_tasks "mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
    --param_ratio_target $1 \
    --use_cache >logs/fwsvd/${model_name}_ratio_${1}.log
}


for ratio in "${ratios[@]}"; do
    echo "ratio: $ratio"
    run_asvd $ratio
done


set +x