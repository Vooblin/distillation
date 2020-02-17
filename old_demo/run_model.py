import tensorflow as tf

import itertools
import json
import os

from absl import app
from absl import flags

import utils

from official.nlp.bert_models import classifier_model
from official.nlp.bert_modeling import BertConfig
from official.nlp.optimization import create_optimizer

utils.disable_logging()
utils.allow_growth_gpu()

flags.DEFINE_string('model_name', default=None, help='Model name')
flags.DEFINE_string('task_name', default=None, help='Task name')
flags.DEFINE_string('tfrecord_dir', default=None, help='Directory for input data in tfrecord format')
flags.DEFINE_string('output_dir', default=None, help='Directory for saved weights')
flags.DEFINE_string('config_path', default=None, help='Path to config file')
flags.DEFINE_integer('max_seq_length', default=128, help='Max sequence length')
flags.DEFINE_integer('batch_size', default=16, help='Batch size')
flags.DEFINE_integer('epochs', default=40, help='Number of epochs')
flags.DEFINE_float('learning_rate', default=1e-4, help='Learning rate')
flags.DEFINE_bool('grid', default=False, help='Use or not grid search')
flags.DEFINE_integer('runs', default=1, help='Number of model runs')
flags.DEFINE_string('logdir', default="./logs/", help='Log directory')

FLAGS = flags.FLAGS

BATCHES = [16, 32, 64, 128, 256]
EPOCHS = [2, 3, 5, 10, 20, 40]
LRS = [1e-4, 5e-5, 3e-5, 2e-5, 1e-5]

def run_simple_model(model_name, task_name, tfrecord_dir, output_dir, config_path, max_seq_length, batch_size, epochs, learning_rate):
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  datasets = utils.get_datasets(task_name, tfrecord_dir, max_seq_length, batch_size)
  train_dataset, test_dataset, valid_dataset, transfer_dataset = datasets
  train_dataset = train_dataset.concatenate(transfer_dataset)

  simple_model = utils.get_model(model_name, config_path, task_name, max_seq_length)
  inputs = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
  model_output = simple_model(inputs)
  output = tf.keras.layers.Activation('softmax')(model_output)
  model = tf.keras.Model(inputs=inputs, outputs=output)

  steps_per_epoch = utils.get_train_steps(task_name, batch_size, epochs)
  validation_steps = utils.get_valid_steps(task_name, batch_size, epochs)

  optimizer = create_optimizer(learning_rate, steps_per_epoch * epochs, 10000)

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

  utils.run_model(model, train_dataset, valid_dataset, steps_per_epoch, validation_steps)
  valid_result = model.evaluate(valid_dataset, verbose=0)
  eval_result = model.evaluate(test_dataset, verbose=0)
  print("Test evaluation: - loss: {0:.4f} - accuracy: {0:.4f}".format(eval_result[0], eval_result[1]))

  output_path = os.path.join(output_dir, '{}_{}_{}_{}_{}_{}.h5'.format(model_name, task_name, max_seq_length, batch_size, epochs, learning_rate))
  model.save_weights(output_path)
  return model, valid_result

def main(_):
  model_name = FLAGS.model_name
  task_name = FLAGS.task_name
  tfrecord_dir = FLAGS.tfrecord_dir
  output_dir = FLAGS.output_dir
  config_path = FLAGS.config_path
  max_seq_length = FLAGS.max_seq_length
  runs = FLAGS.runs
  batches = BATCHES if FLAGS.grid else [FLAGS.batch_size]
  epochs = EPOCHS if FLAGS.grid else [FLAGS.epochs]
  lrs = LRS if FLAGS.grid else [FLAGS.learning_rate]

  grid_results = dict()
  for batch_size, epoch, lr in itertools.product(batches, epochs, lrs):
    results = list()
    for run in range(runs):
      try:
        result = run_simple_model(model_name, task_name, tfrecord_dir, output_dir, config_path, max_seq_length, batch_size, epoch, lr)
        results.append(result[1])
      except Exception as e:
        print("ERROR: On batch_size={}, epochs={}, learning_rate={}, iteration={} -- {}".format(batch_size, epoch, lr, run, e))
    results = list(zip(*results))
    results = [sum(elem) / len(elem) for elem in results]
    grid_results["{}_{}_{}".format(batch_size, epoch, lr)] = {'loss': results[0], 'accuracy': results[1]}

  logdir = FLAGS.logdir
  if not os.path.exists(logdir):
    os.mkdir(logdir)
  logpath = os.path.join(logdir, '{}_{}_{}.json'.format(model_name, task_name, max_seq_length))
  with open(logpath, 'w') as f:
    json.dump(grid_results, f)

if __name__ == '__main__':
  app.run(main)
