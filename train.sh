#!/usr/bin/env bash

export PYTHONPATH=/Users/ruanjiaqiang/Desktop/programing/python/chinese-word-segmentation:$PYTHONPATH

set -ex
PKUTRAIN="/Users/ruanjiaqiang/Downloads/中文分词数据集/PKU&MSRA/icwb2-data/training/pku_training.utf8"
PKUTEST="/Users/ruanjiaqiang/Downloads/中文分词数据集/PKU&MSRA/icwb2-data/testing/pku_test.utf8"
PKUGOLD="/Users/ruanjiaqiang/Downloads/中文分词数据集/PKU&MSRA/icwb2-data/gold/pku_test_gold.utf8"

INPUT=$PKUTRAIN
VALIDATION=$PKUTEST
REFERENCE=$PKUGOLD
DEVICE=1

python /Users/ruanjiaqiang/Desktop/programing/python/chinese-word-segmentation/train.py \
    --input $INPUT \
    --output debug \
    --model 1 \
    --validation $VALIDATION \
    --reference $REFERENCE \
    --parameters device_list=[$DEVICE]
