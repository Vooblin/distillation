import json
import logging
import os
import random
import sys

from absl import app
from absl import flags
import tensorflow as tf

from dataset_processors import get_processor
from official.nlp.bert import tokenization
from official.nlp.bert.classifier_data_lib import file_based_convert_examples_to_features

flags.DEFINE_string('task_name', default=None, help='Task name')
flags.DEFINE_string('data_dir', default=None, help='Directory for input data.')
flags.DEFINE_string('vocab_path', default=None, help='Path to vocabulary file')
flags.DEFINE_string('output_dir', default=None, help='Output dir for TF records.')
flags.DEFINE_integer("max_seq_length", default=128, help='Max sequence length')
flags.DEFINE_bool('overwrite_data', default=False, help='If False, will use cached data if available.')

FLAGS = flags.FLAGS

#logging.disable(logging.INFO)

def get_data_from_processor(data_dir, processor):
  train_examples = processor.get_train_examples(data_dir)
  test_examples = processor.get_dev_examples(data_dir)
  label_list = processor.get_labels()

  random.shuffle(train_examples)
  random.shuffle(test_examples)

  train_split = int(len(train_examples) * 0.45)
  transfer_split = int(len(train_examples) * 0.9)
  transfer_examples = train_examples[train_split:transfer_split]
  valid_examples = train_examples[transfer_split:]
  train_examples = train_examples[:train_split]

  return (train_examples, test_examples, valid_examples, transfer_examples), label_list

def create_tfrecord_data(task_name, data_dir, vocab_path, output_dir, max_seq_length=128, overwrite_data=False):
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  paths = []
  for name in ['train', 'test', 'valid', 'transfer']:
    output_path = os.path.join(output_dir, '{}_{}.tfrecord'.format(task_name, name))
    paths.append(output_path)

  processor = get_processor(task_name)
  all_examples, label_list = get_data_from_processor(data_dir, processor)

  meta_data = {
      "task_name": task_name,
      "num_classes": len(processor.get_labels()),
      "train_data_size": len(all_examples[0]),
      "test_data_size": len(all_examples[1]),
      "valid_data_size": len(all_examples[2]),
      "transfer_data_size": len(all_examples[3]),
      "max_seq_length": max_seq_length,
  }
  meta_data_path = os.path.join(output_dir, 'meta_{}.json'.format(task_name))
  with tf.io.gfile.GFile(meta_data_path, "w") as f:
    f.write(json.dumps(meta_data, indent=4) + "\n")

  tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=True)
  for path, examples in zip(paths, all_examples):
    if not (os.path.exists(path) and overwrite_data == False):
      file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, path)

def main(_):
  task_name = FLAGS.task_name
  data_dir = FLAGS.data_dir
  vocab_path = FLAGS.vocab_path
  output_dir = FLAGS.output_dir
  max_seq_length = FLAGS.max_seq_length
  overwrite_data = FLAGS.overwrite_data

  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  if os.path.isdir(data_dir) and os.path.exists(vocab_path) and os.path.isdir(output_dir):
    create_tfrecord_data(task_name, data_dir, vocab_path, output_dir, max_seq_length, overwrite_data)
  else:
    raise ValueError('data_dir and output_dir should be directories and vocab_path should exist')

if __name__ == '__main__':
  app.run(main)
