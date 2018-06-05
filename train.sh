#!/usr/bin/env bash

#PKUTRAIN="/Users/ruanjiaqiang/Downloads/中文分词数据集/PKU&MSRA/icwb2-data/training/pku_training.utf8"
#PKUTEST="/Users/ruanjiaqiang/Downloads/中文分词数据集/PKU&MSRA/icwb2-data/testing/pku_test.utf8"
#PKUGOLD="/Users/ruanjiaqiang/Downloads/中文分词数据集/PKU&MSRA/icwb2-data/gold/pku_test_gold.utf8"

export LD_LIBRARY_PATH=/data/disk1/private/zjc/cuda/cuda-9.0/lib64:$LD_LIBRARY_PATH
export PATH=/data/disk1/private/zjc/cuda/cuda-9.0/bin:$PATH
export PYTHONPATH=/Users/ruanjiaqiang/Desktop/programing/python/chinese-word-segmentation:$PYTHONPATH

set -ex
PKUTRAIN="/home/rjq/project/cws/PKU&MSRA/icwb2-data/training/pku_training.utf8"
PKUTEST="/home/rjq/project/cws/PKU&MSRA/icwb2-data/testing/pku_test.utf8"
PKUGOLD="/home/rjq/project/cws/PKU&MSRA/icwb2-data/gold/pku_test_gold.utf8"

INPUT=$PKUTRAIN
VALIDATION=$PKUTEST
REFERENCE=$PKUGOLD
DEVICE=6

python /home/rjq/project/chinese-word-segmentation/train.py \
    --input $INPUT \
    --output train \
    --model 1 \
    --validation $VALIDATION \
    --reference $REFERENCE \
    --parameters device_list=[$DEVICE],batch_size=128,validate_steps=20
