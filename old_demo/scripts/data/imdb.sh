#!/bin/sh

script_dir=$(dirname $(readlink -f $0))
demo_dir="$script_dir/../.."
export PYTHONPATH="$PYTHONPATH:$demo_dir/models/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu"

datasets_dir="$demo_dir/datasets"
if [ ! -d $datasets_dir ]
then
  mkdir $datasets_dir
fi

task_name='imdb'
data_dir="$datasets_dir/imdb"
if [ ! -d $data_dir ]
then
  wget --directory-prefix=$datasets_dir "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  tar --directory=$datasets_dir -xf "$datasets_dir/aclImdb_v1.tar.gz"
  mv "$datasets_dir/aclImdb" "$datasets_dir/imdb"
  rm "$datasets_dir/aclImdb_v1.tar.gz"
fi

vocab_dir="$demo_dir/bert"
vocab_path="$vocab_dir/vocab.txt"
if [ ! -d $vocab_dir ]
then
  wget --directory-prefix=$demo_dir  "https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/cased_L-12_H-768_A-12.tar.gz"
  tar --directory=$demo_dir -xf "$demo_dir/cased_L-12_H-768_A-12.tar.gz"
  mv "$demo_dir/cased_L-12_H-768_A-12" "$demo_dir/bert"
  rm "$demo_dir/cased_L-12_H-768_A-12.tar.gz"
fi

output_dir="$demo_dir/tfrecords"

echo "Start tfrecord generation"

python3 "$demo_dir/dataset2tfrecord.py" \
  --task_name=$task_name \
  --data_dir=$data_dir \
  --vocab_path=$vocab_path \
  --output_dir=$output_dir

echo "End tfrecord generation"
