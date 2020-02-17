import csv
import json
import logging
import os
import random
import sys

from absl import app
from absl import flags

from official.nlp.bert import tokenization
from official.nlp.bert.classifier_data_lib import MnliProcessor
from official.nlp.bert.classifier_data_lib import file_based_convert_examples_to_features
from official.nlp.xlnet.preprocess_classification_data import ImdbProcessor, Yelp5Processor

flags.DEFINE_string('vocab_path', None,
                    'Path to vocabulary file')

FLAGS = flags.FLAGS

max_int = sys.maxsize
csv.field_size_limit(max_int)

logging.disable(logging.INFO)

def create_tfrecord_data(task_name, data_dir, vocab_path, output_dir, max_seq_length=128, overwrite_data=False):
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  paths = []
  for name in ['train', 'test', 'valid', 'transfer']:
    output_path = os.path.join(output_dir, '{}_{}_{}.tfrecord'.format(task_name, max_seq_length, name))
    paths.append(output_path)
  processor = get_processor(task_name)
  all_examples, label_list = processor(data_dir)
  tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_path, do_lower_case=True)
  for path, examples in zip(paths, all_examples):
    if not (os.path.exists(path) and overwrite_data == False):
      file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, path)


def get_processor(task_name):
  processors = {
      'imdb': _imdb_processor,
      'mnli': _mnli_processor,
      'yelp': _yelp_processor,
  }
  if task_name in processors.keys():
    return processors[task_name]
  else:
    raise ValueError("task_name is '{}' but it should be in {}".format(task_name, list(processors.keys()))) 

def _imdb_processor(data_dir):
  processor = ImdbProcessor()
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

def _mnli_processor(data_dir):
  class MyMnliProcessor(MnliProcessor):
    @classmethod
    def _read_tsv(cls, input_file):
      with open(input_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        num_elems = len(header)
        elems = []
        for line in reader:
          for elem in line:
            elems.extend(elem.replace('\n', '\t').split('\t'))
        lines = [elems[i * num_elems: (i + 1) * num_elems] for i in range(len(elems) // num_elems)]
      return [header] + lines

  processor = MyMnliProcessor()
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

def _yelp_processor(data_dir):
  class MyYelp5Processor(Yelp5Processor):
    def __init__(self):
      output_path = os.path.join(data_dir, 'data.csv')
      input_path = os.path.join(data_dir, 'yelp_academic_dataset_review.json')
      with open(input_path, 'r') as f:
        reviews = []
        for line in f:
          reviews.append(json.loads(line))
      data = [(str(int(float(review['stars']))), review['text']) for review in reviews]
      with open(output_path, 'w') as f:
        writer = csv.writer(f)
        for line in data:
          writer.writerow(line)

    def get_examples(self, data_dir):
      return self._create_examples(os.path.join(data_dir, 'data.csv'))

  processor = MyYelp5Processor()
  examples = processor.get_examples(data_dir)
  label_list = processor.get_labels()

  random.shuffle(examples)
  train_split = int(len(examples) * 0.4)
  transfer_split = int(len(examples) * 0.8)
  valid_split = int(len(examples) * 0.9)
  train_examples = examples[:train_split]
  transfer_examples = examples[train_split:transfer_split]
  valid_examples = examples[transfer_split:valid_split]
  test_examples = examples[valid_split:]
  return (train_examples, test_examples, valid_examples, transfer_examples), label_list


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
