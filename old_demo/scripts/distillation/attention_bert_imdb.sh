#!/bin/sh

script_dir=$(dirname $(readlink -f $0))
demo_dir="$script_dir/../.."
export PYTHONPATH="$PYTHONPATH:$demo_dir/models/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu"

task_name='imdb'
student_model_name="attention"
teacher_model_name="bert"
student_config_path="$demo_dir/configs/attention.json"
teacher_config_path="$demo_dir/bert/bert_config.json"
weights_dir="$demo_dir/weights"
tfrecord_dir="$demo_dir/tfrecords"

result_dir="$demo_dir/results"
if [ ! -d $result_dir ]
then
  mkdir $result_dir
fi
result_path="$result_dir/distil_attention_bert_imdb.txt"

echo "Start distillation from BERT Base to $student_model_name"

python3 "$demo_dir/distillation.py" \
  --task_name=$task_name \
  --student_model_name=$student_model_name \
  --teacher_model_name=$teacher_model_name \
  --student_config_path=$student_config_path \
  --teacher_config_path=$teacher_config_path \
  --weights_dir=$weights_dir \
  --tfrecord_dir=$tfrecord_dir | \
tee $result_path

echo "End distillation from BERT Base to $student_model_name"
