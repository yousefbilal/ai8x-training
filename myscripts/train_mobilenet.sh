#!/bin/bash

heights=(25 35 45 55 65)

seeds=(1 2 3 4 5 6 7 8 9 10)

for height in "${heights[@]}"; do
    echo "Starting training for height $height..."

    for seed in "${seeds[@]}"; do
        echo "Training with seed $seed for height $height"
        python train.py \
            --lr 0.001 \
            --optimizer Adam \
            --epochs 30 \
            --deterministic \
            --seed $seed \
            --model mymobilenet \
            --dataset desert_${height} \
            --confusion \
            --pr-curves \
            --device MAX78002 \
            --qat-policy policies/qat_policy_cd.yaml \
            --batch-size 16 \
            --use-bias \
            --enable-tensorboard \
            -n mobilenet_${height}_${seed} \
	    -p 50
    done

    echo "Training completed for height $height."
done

echo "All training tasks completed!"
