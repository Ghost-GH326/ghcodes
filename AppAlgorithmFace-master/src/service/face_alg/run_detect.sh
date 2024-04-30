#!/bin/bash
python main.py \
    --path '/data1/dataset/glintface/train_msra/files/align_test100.json' \
    --info-path '/data1/dataset/glintface/train_msra/files/train_msra_aligned_detect.info' \
    --process-num 2 \
    --line-num 100 \
    --task-name 'detect' \
    --batch-size 8 \
    --network 'net3' \
    --model-path './model/retina-0000.params' \
    --input-size '640,640' \
    --threshold 0.8 \
    --gpus '0' \
