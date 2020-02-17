import tensorflow as tf

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

flags.DEFINE_string('task_name', default=None, help='Task name')
flags.DEFINE_string('tfrecord_dir', default=None, help='Directory for input data in tfrecord format')
flags.DEFINE_string('output_dir', default=None, help='Directory for saved weights')
flags.DEFINE_string('config_path', default=None, help='Path to config file')
flags.DEFINE_string('ckpt_path', default=None, help='Path to checkpoint file')
flags.DEFINE_integer('max_seq_length', default=128, help='Max sequence length')
flags.DEFINE_integer('batch_size', default=8, help='Batch size')

FLAGS = flags.FLAGS

def run_bert(task_name, tfrecord_dir, output_dir, config_path, ckpt_path, max_seq_length, batch_size):
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  datasets = utils.get_datasets(task_name, tfrecord_dir, max_seq_length, batch_size)
  train_dataset, test_dataset, valid_dataset, transfer_dataset = datasets

  bert_model = utils.get_model('bert', config_path, task_name, max_seq_length)
  output = tf.keras.layers.Activation('softmax')(bert_model.outputs[0])
  model = tf.keras.Model(inputs=bert_model.inputs, outputs=output)
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(ckpt_path)

  num_train_steps = utils.get_train_steps(task_name)
  num_valid_steps = utils.get_valid_steps(task_name)
  num_epochs = utils.get_epochs()

  optimizer = create_optimizer(2e-5, num_train_steps * num_epochs, num_train_steps * num_epochs // 10)

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

  utils.run_model(model, train_dataset, valid_dataset, task_name)
  model.evaluate(test_dataset, verbose=2)

  output_path = os.path.join(output_dir, 'bert_{}_{}.h5'.format(task_name, max_seq_length))
  model.save_weights(output_path)

def main(_):
  task_name = FLAGS.task_name
  tfrecord_dir = FLAGS.tfrecord_dir
  output_dir = FLAGS.output_dir
  config_path = FLAGS.config_path
  ckpt_path = FLAGS.ckpt_path
  max_seq_length = FLAGS.max_seq_length
  batch_size = FLAGS.batch_size

  run_bert(task_name, tfrecord_dir, output_dir, config_path, ckpt_path, max_seq_length, batch_size)

if __name__ == '__main__':
  app.run(main)
