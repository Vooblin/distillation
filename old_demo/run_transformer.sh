#!/bin/sh

python3 run_model.py --model_name transformer --task_name imdb --tfrecord_dir /workspace/murygin/KD/demo/tfrecords/ --output_dir /workspace/murygin/KD/demo/weights/ --config_path /workspace/murygin/KD/demo/configs/transformer.json --grid True --runs 1
python3 run_model.py --model_name transformer --task_name cola --tfrecord_dir /workspace/murygin/KD/demo/tfrecords/ --output_dir /workspace/murygin/KD/demo/weights/ --config_path /workspace/murygin/KD/demo/configs/transformer.json --grid True --runs 1
