import tensorflow as tf

import json
import os

from absl import app
from absl import flags

import utils

from official.nlp.bert_models import classifier_model
from official.nlp.bert_modeling import BertConfig
from official.nlp.optimization import create_optimizer

tf.compat.v1.disable_resource_variables()

utils.disable_logging()
utils.allow_growth_gpu()

flags.DEFINE_string('task_name', default=None, help='Task name')
flags.DEFINE_string('weight_dir', default=None, help='Directory for model weights')
flags.DEFINE_string('output_dir', default=None, help='Directory for tflite models')
flags.DEFINE_string('config_path', default=None, help='Path to config file')
flags.DEFINE_integer('max_seq_length', default=128, help='Max sequence length')

FLAGS = flags.FLAGS

def convert_bert(task_name, weight_dir, output_dir, config_path, max_seq_length):
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  weight_path = os.path.join(weight_dir, 'bert_{}_{}.h5'.format(task_name, max_seq_length))

  bert_model = utils.get_model('bert', config_path, task_name, max_seq_length)
  output = tf.keras.layers.Activation('softmax')(bert_model.outputs[0])
  model = tf.keras.Model(inputs=bert_model.inputs, outputs=output)
  model.load_weights(weight_path)

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  output_path = os.path.join(output_dir, 'bert_{}_{}.tflite'.format(task_name, max_seq_length))
  with open(output_path, "wb") as f:
      f.write(tflite_model)


def main(_):
  task_name = FLAGS.task_name
  weight_dir = FLAGS.weight_dir
  output_dir = FLAGS.output_dir
  config_path = FLAGS.config_path
  max_seq_length = FLAGS.max_seq_length

  convert_bert(task_name, weight_dir, output_dir, config_path, max_seq_length)

if __name__ == '__main__':
  app.run(main)
