#!/usr/bin/env bash

PKUDICT=/home/rjq/project/cws/PKU&MSRA/icwb2-data/gold/pku_training_words.utf8
PKUGOLD=/home/rjq/project/cws/PKU&MSRA/icwb2-data/gold/pku_test_gold.utf8
MSRADICT=/home/rjq/project/cws/PKU&MSRA/icwb2-data/gold/msr_training_words.utf8
MSRAGOLD=/home/rjq/project/cws/PKU&MSRA/icwb2-data/gold/msr_test_gold.utf8

DICT=$PKUDICT
GOLD=$PKUGOLD

for file in `ls`
do
echo $file
perl /home/rjq/project/chinese-word-segmentation/score $DICT $GOLD $file > $file-score
tail $file-score
done