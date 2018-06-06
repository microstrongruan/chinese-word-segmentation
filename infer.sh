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

MSRATRAIN="/home/rjq/project/cws/PKU&MSRA/icwb2-data/training/msr_training.utf8"
MSRATEST="/home/rjq/project/cws/PKU&MSRA/icwb2-data/testing/msr_test.utf8"
MSRAGOLD="/home/rjq/project/cws/PKU&MSRA/icwb2-data/gold/msr_test_gold.utf8"

CTB6TRAIN="/home/rjq/project/cws/CTB6/trainSegmenter/data/ctb6.train.seg"
CTB6TEST="/home/rjq/project/cws/CTB6/trainSegmenter/data/ctb6.test.unseg"
CTB6GOLD="/home/rjq/project/cws/CTB6/trainSegmenter/data/ctb6.test.seg"

INPUT=$PKUTEST
DEVICE=5
CHECKPOINTS=debug/model.ckpt-
JSON=train/params.json

python /home/rjq/project/chinese-word-segmentation/infer.py \
    --input $INPUT \
    --output test \
    --checkpoints $CHECKPOINTS \
    --json $JSON \
    --model 1 \
    --parameters device_list=[$DEVICE],batch_size=1,search_policy=viterbi