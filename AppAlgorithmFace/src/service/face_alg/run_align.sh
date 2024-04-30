#!/bin/bash
python main.py \
    --path '/data1/dataset/glintface/train_msra/files/align_test100.json' \
    --info-path '/data1/dataset/glintface/train_msra/files/train_msra_aligned_100.info' \
    --process-num 2 \
    --line-num 100 \
    --align-shape '112,112' \
    --task-name 'align' \
    --save-root '/data1/dataset/glintface/train_msra/files/test_imgs'
