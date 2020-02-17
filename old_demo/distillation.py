import tensorflow as tf
from tensorflow.keras.layers import Activation

import os

from birnn import BiLSTMModel, BiGRUModel
from attention import AttentionModel
from transformer import TransformerModel

import utils

from absl import app
from absl import flags

from official.nlp.bert_models import classifier_model
from official.nlp.bert_modeling import BertConfig

utils.disable_logging()
utils.allow_growth_gpu()

flags.DEFINE_string('student_model_name', default=None, help='Student model_name')
flags.DEFINE_string('teacher_model_name', default='bert', help='Teacher model_name')
flags.DEFINE_string('student_config_path', default=None, help='Student model config path')
flags.DEFINE_string('teacher_config_path', default=None, help='Teacher model config path')
flags.DEFINE_string('weights_dir', default=None, help='Directory with weights of saved models')
flags.DEFINE_string('task_name', default=None, help='Task name')
flags.DEFINE_string('tfrecord_dir', default=None, help='Directory with datasets in tfrecord format')
flags.DEFINE_integer('max_seq_length', default=128, help='Max sequence length')
flags.DEFINE_integer('batch_size', default=8, help='Batch size')

FLAGS = flags.FLAGS

def distillation(teacher_model, student_model, train_dataset, valid_dataset, test_dataset, task_name, temp=1.0, a=0.5, b=0.5):

  def update_train_dataset(inputs, labels):
    teacher_logits = teacher_model(inputs)
    teacher_predictions = Activation('softmax')(teacher_logits / temp)
    return (inputs, (labels, teacher_predictions))

  train_dataset = train_dataset.map(update_train_dataset)
  valid_dataset = valid_dataset.map(update_train_dataset)

  inputs = student_model.inputs
  logits = student_model(inputs)
  usual_output = Activation('softmax', name='hard')(logits)
  transfer_output = Activation('softmax', name='soft')(logits / temp)
  student_train_model = tf.keras.Model(inputs=inputs, outputs=(usual_output, transfer_output))
  student_eval_model = tf.keras.Model(inputs=inputs, outputs=usual_output)
  
  student_train_model.compile(loss=('sparse_categorical_crossentropy', 'categorical_crossentropy'),
                              loss_weights=[a, b],
                              optimizer='adam',
                              metrics=['accuracy'])

  student_eval_model.compile(loss='sparse_categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

  utils.run_model(student_train_model, train_dataset, valid_dataset, task_name)
  student_eval_model.set_weights(student_train_model.get_weights())
  student_eval_model.evaluate(test_dataset, verbose=2)
  return student_eval_model

def main(_):
  student_model_name = FLAGS.student_model_name
  teacher_model_name = FLAGS.teacher_model_name
  student_config_path = FLAGS.student_config_path
  teacher_config_path = FLAGS.teacher_config_path
  weights_dir = FLAGS.weights_dir
  task_name = FLAGS.task_name
  tfrecord_dir = FLAGS.tfrecord_dir
  max_seq_length = FLAGS.max_seq_length
  batch_size = FLAGS.batch_size

  student_model = utils.get_model(student_model_name, student_config_path, task_name, max_seq_length)
  inputs = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
  output = student_model(inputs)
  student_model = tf.keras.Model(inputs=inputs, outputs=output)

  teacher_model = utils.get_model(teacher_model_name, teacher_config_path, task_name, max_seq_length)
  teacher_weights_path = os.path.join(weights_dir, "{}_{}_{}.h5".format(teacher_model_name, task_name, max_seq_length))
  teacher_model.load_weights(teacher_weights_path)

  datasets = utils.get_datasets(task_name, tfrecord_dir, max_seq_length, batch_size)
  train_dataset, test_dataset, valid_dataset, transfer_dataset = datasets
  result_model = distillation(teacher_model, student_model, transfer_dataset, valid_dataset, test_dataset, task_name, 1.0, 0.5, 0.5)

  save_path = os.path.join(weights_dir, "distil_{}_{}_{}_{}.h5".format(student_model_name, teacher_model_name, task_name, max_seq_length))
  result_model.save_weights(save_path)

if __name__ == "__main__":
  app.run(main)
