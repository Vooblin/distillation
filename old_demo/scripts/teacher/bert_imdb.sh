#!/bin/sh

script_dir=$(dirname $(readlink -f $0))
demo_dir="$script_dir/../.."
export PYTHONPATH="$PYTHONPATH:$demo_dir/models/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu"
"$demo_dir/scripts/data/imdb.sh"

task_name='imdb'
tfrecord_dir="$demo_dir/tfrecords"
output_dir="$demo_dir/weights"
config_path="$demo_dir/bert/bert_config.json"
ckpt_path="$demo_dir/bert/bert_model.ckpt"
result_dir="$demo_dir/results"
if [ ! -d $result_dir ]
then
  mkdir $result_dir
fi
result_path="$result_dir/bert_imdb.txt"

echo "Start training BERT Base"

python3 "$demo_dir/run_bert.py" \
  --task_name=$task_name \
  --tfrecord_dir=$tfrecord_dir \
  --output_dir=$output_dir \
  --config_path=$config_path \
  --ckpt_path=$ckpt_path | \
tee $result_path

echo "End training BERT Base"
