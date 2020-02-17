#!/bin/sh

script_dir=$(dirname $(readlink -f $0))
models_dir="$script_dir/models"
mkdir -p $models_dir
#git clone "https://github.com/tensorflow/models.git" $models_dir
cd "$models_dir"
#git checkout f52b8c9
sed -i 's/use_bias=True/use_bias=False/g' "./official/nlp/bert_modeling.py"
