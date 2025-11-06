#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=2,3


# 0.1 0.2 0.3 0.4 0.5 0.6
ratios=(0.9 0.8 0.7 0.6 0.5 0.4 0.2)
# ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
model="meta-llama/Llama-2-13b-hf"
model_name=$(echo "$model" | awk -F'/' '{print $2}')

run_asvd()
{
    python asvd.py \
    --model_id ${model} \
    --act_aware \
    --alpha 0.5 \
    --n_calib_samples 256 \
    --scaling_method abs_mean \
    --param_ratio_target $1 \
    --eval_tasks "mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
    --use_cache >logs/asvd/${model_name}_ratio_${1}.log
}


for ratio in "${ratios[@]}"; do
    echo "ratio: $ratio"
    run_asvd $ratio
done


set +x