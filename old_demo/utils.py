import tensorflow as tf

import json
import logging
import os

from attention import AttentionModel
from birnn import BiLSTMModel, BiGRUModel
from transformer import TransformerModel

from official.nlp.bert_modeling import BertConfig
from official.nlp.bert_models import classifier_model
from official.nlp.bert.input_pipeline import create_classifier_dataset

def disable_logging():
  logging.disable(logging.INFO)
  logging.disable(logging.WARNING)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def allow_growth_gpu():
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  config.log_device_placement = True
  sess = tf.compat.v1.Session(config=config)
  tf.compat.v1.keras.backend.set_session(sess)

def get_epochs():
  return 100

def get_labels(task_name):
  labels = {
      'imdb': 2,
      'mnli': 3,
      'yelp': 5,
  }
  return labels[task_name]

def get_train_steps(task_name, batch_size=8, epochs=100):
  steps = {
      'imdb': (22500 * epochs) // (batch_size * 100),
      'mnli': 10000,
      'yelp': 100000,
  }
  return steps[task_name]

def get_valid_steps(task_name, batch_size=8, epochs=100):
  steps = {
      'imdb': 2500 // batch_size,
      'mnli': 1000,
      'yelp': 1000,
  }
  return steps[task_name]

def get_datasets(task_name, tfrecord_dir, max_seq_length, batch_size):
  datasets = []
  for name, is_training in zip(['train', 'test', 'valid', 'transfer'], [True, False, False, True]):
    path = os.path.join(tfrecord_dir, '{}_{}_{}.tfrecord'.format(task_name, max_seq_length, name))
    dataset = create_classifier_dataset(path, max_seq_length, batch_size, is_training=is_training)
    datasets.append(dataset)
  return datasets

def _BertModel(config):
  num_labels = config['num_labels']
  max_seq_length = config['max_seq_length']
  model = classifier_model(BertConfig.from_dict(config), tf.float32, num_labels, max_seq_length)[0]
  return model

def get_model(model_name, config_path, task_name, max_seq_length):
  with open(config_path, 'r') as f:
    config = json.load(f)
  config['num_labels'] = get_labels(task_name)
  config['max_seq_length'] = max_seq_length
  models = {
      'bilstm': BiLSTMModel,
      'bigru': BiGRUModel,
      'attention': AttentionModel,
      'transformer': TransformerModel,
      'shared_transformer': SharedTransformerModel,
      'bert': _BertModel}
  model = models[model_name](config)
  return model

def run_model(model, train_dataset, valid_dataset, steps_per_epoch, validation_steps):
  callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
  model.fit(train_dataset, epochs=100, steps_per_epoch=steps_per_epoch, validation_data=valid_dataset, callbacks=callbacks, validation_steps=validation_steps, verbose=1)
