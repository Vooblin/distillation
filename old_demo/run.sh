#!/bin/sh

set -e -x

for filename in /workspace/murygin/datasets/unlabeled/bert_pretraining_texts/*; do
    python3 create_distilling_data.py --input_file=$filename --output_file="$filename.tfrecord" --vocab_file=/workspace/murygin/bert_tflite/demo/bert/vocab.txt --do_lower_case=True --max_seq_length=128 --random_seed=10810
done
