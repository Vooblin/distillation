import functools
import json
import math
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from official.nlp import optimization
from official.utils.misc import distribution_utils
from official.modeling.model_training_utils import run_customized_training_loop
from official.nlp.bert import input_pipeline

from custom_bert_modeling import CustomBertConfig, get_custom_bert_model

def allow_growth_gpu():
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  config.log_device_placement = True
  sess = tf.compat.v1.Session(config=config)
  tf.compat.v1.keras.backend.set_session(sess)

def limit_gpu_memory(limit, idx=0):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(gpus[idx],
      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)])

limit_gpu_memory(6.0 * 1024.0)

_SUMMARY_TXT = 'training_summary.txt'
_MIN_SUMMARY_STEPS = 10

flags.DEFINE_string('task_name', default=None, help='Task name')
flags.DEFINE_string('data_dir', default=None, help='Directory for input data in tfrecord format')
flags.DEFINE_string('output_dir', default=None, help='Directory for saved weights')
flags.DEFINE_string('config_path', default=None, help='Path to config file')
flags.DEFINE_string('teacher_config_path', default=None, help='Path to config file of teacher model')
flags.DEFINE_string('ckpt_path', default=None, help='Path to checkpoint file')
flags.DEFINE_string('teacher_ckpt_path', default=None, help='Path to checkpoint file of teacher model')
flags.DEFINE_integer('max_seq_length', default=128, help='Max sequence length')
flags.DEFINE_integer('batch_size', default=8, help='Batch size')
flags.DEFINE_integer('epochs', default=3, help='Number of training epochs')
flags.DEFINE_float('learning_rate', default=5e-5, help='Initial learning rate')

FLAGS = flags.FLAGS

def get_dataset(task_name, data_dir, dataset_type='train', max_seq_length=128, batch_size=8, add_transfer=True):
  is_training = False
  if dataset_type in ['train', 'transfer']:
    is_training = True

  path = os.path.join(data_dir, '{}_{}.tfrecord'.format(task_name, dataset_type))
  dataset = input_pipeline.create_classifier_dataset(path, max_seq_length, batch_size, is_training=is_training)

  if dataset_type == 'train' and add_transfer:
    transfer_path = os.path.join(data_dir, '{}_{}.tfrecord'.format(task_name, 'transfer'))
    transfer_dataset = input_pipeline.create_classifier_dataset(
        transfer_path, max_seq_length, batch_size, is_training=True)
    dataset = dataset.concatenate(transfer_dataset)

  return dataset

def classifier_model(config, num_labels, max_seq_length, distillation=False):
  input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

  model = get_custom_bert_model(input_word_ids, input_mask, input_type_ids, config=config, name='bert')
  pooled_output = model.outputs[0]

  initializer = tf.keras.initializers.TruncatedNormal(
      stddev=config.initializer_range)

  output = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)(pooled_output)
  output = tf.keras.layers.Dense(num_labels, kernel_initializer=initializer, name='output')(output)
  if distillation:
    length = config.num_hidden_groups * config.inner_group_num
    output = (model.outputs[0],
              model.outputs[1],
              model.outputs[2],
              model.outputs[3: 3 + length],
              model.outputs[3 + length: 3 + 2 * length],
              output)
  return tf.keras.Model(
      inputs={
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids
      },
      outputs=output), model

def _save_checkpoint(checkpoint, model_dir, checkpoint_prefix):
  """Saves model to with provided checkpoint prefix."""

  checkpoint_path = os.path.join(model_dir, checkpoint_prefix)
  saved_path = checkpoint.save(checkpoint_path)
  logging.info('Saving model as TF checkpoint: %s', saved_path)
  return

def _get_input_iterator(input_fn, strategy):
  """Returns distributed dataset iterator."""

  # When training with TPU pods, datasets needs to be cloned across
  # workers. Since Dataset instance cannot be cloned in eager mode, we instead
  # pass callable that returns a dataset.
  input_data = input_fn()
  if callable(input_data):
    iterator = iter(
        strategy.experimental_distribute_datasets_from_function(input_data))
  else:
    iterator = iter(strategy.experimental_distribute_dataset(input_data))
  return iterator


def _float_metric_value(metric):
  """Gets the value of a float-value keras metric."""
  return metric.result().numpy().astype(float)


def steps_to_run(current_step, steps_per_epoch, steps_per_loop):
  """Calculates steps to run on device."""
  if steps_per_loop <= 0:
    raise ValueError('steps_per_loop should be positive integer.')
  if steps_per_loop == 1:
    return steps_per_loop
  remainder_in_epoch = current_step % steps_per_epoch
  if remainder_in_epoch != 0:
    return min(steps_per_epoch - remainder_in_epoch, steps_per_loop)
  else:
    return steps_per_loop


def write_txt_summary(training_summary, summary_dir):
  """Writes a summary text file to record stats."""
  summary_path = os.path.join(summary_dir, _SUMMARY_TXT)
  with tf.io.gfile.GFile(summary_path, 'wb') as f:
    logging.info('Training Summary: \n%s', str(training_summary))
    f.write(json.dumps(training_summary, indent=4))

def run_finetuning_distilling_loop(
    strategy=None,
    teacher_model_fn=None,
    student_model_fn=None,
    loss_fn=None,
    model_dir=None,
    train_input_fn=None,
    eval_input_fn=None,
    eval_steps=None,
    steps_per_epoch=None,
    steps_per_loop=1,
    epochs=1,
    teacher_init_checkpoint=None,
    student_init_checkpoint=None,
    metric_fn=None):

  required_arguments = [
      strategy, teacher_model_fn, student_model_fn, loss_fn, model_dir, steps_per_epoch, train_input_fn
  ]
  if [arg for arg in required_arguments if arg is None]:
    raise ValueError('`strategy`, `model_fn`, `loss_fn`, `model_dir`, '
                     '`steps_per_loop` and `steps_per_epoch` are required '
                     'parameters.')
  if steps_per_loop > steps_per_epoch:
    logging.error(
        'steps_per_loop: %d is specified to be greater than '
        ' steps_per_epoch: %d, we will use steps_per_epoch as'
        ' steps_per_loop.', steps_per_loop, steps_per_epoch)
    steps_per_loop = steps_per_epoch
  assert tf.executing_eagerly()

  total_training_steps = steps_per_epoch * epochs

  train_iterator = _get_input_iterator(train_input_fn, strategy)

  with distribution_utils.get_strategy_scope(strategy):
    teacher_model, teacher_core = teacher_model_fn()
    student_model, student_core = student_model_fn()
    if not hasattr(student_model, 'optimizer'):
      raise ValueError('User should set optimizer attribute to student_model.')
    optimizer = student_model.optimizer

    if teacher_init_checkpoint:
      logging.info(
          'Checkpoint file %s found and restoring from '
          'initial checkpoint for core model.', teacher_init_checkpoint)
      checkpoint = tf.train.Checkpoint(model=teacher_model)
      checkpoint.restore(teacher_init_checkpoint).expect_partial()
      logging.info('Loading from checkpoint file completed')
    if student_init_checkpoint:
      logging.info(
          'Checkpoint file %s found and restoring from '
          'initial checkpoint for core model.', student_init_checkpoint)
      checkpoint = tf.train.Checkpoint(model=student_core)
      checkpoint.restore(student_init_checkpoint).expect_partial()
      logging.info('Loading from checkpoint file completed')

    eval_metrics = [metric_fn()] if metric_fn else []
    train_loss_metric = tf.keras.metrics.Mean(
        'training_loss', dtype=tf.float32)
    train_metrics = [
        metric.__class__.from_config(metric.get_config())
        for metric in eval_metrics
    ]

    summary_dir = os.path.join(model_dir, 'summaries')
    eval_summary_writer = tf.summary.create_file_writer(
        os.path.join(summary_dir, 'eval'))
    if steps_per_loop >= _MIN_SUMMARY_STEPS:
      train_summary_writer = tf.summary.create_file_writer(
          os.path.join(summary_dir, 'train'))
    else:
      train_summary_writer = None

    training_vars = student_model.trainable_variables

    def _replicated_step(inputs):

      inputs, labels = inputs
      with tf.GradientTape() as tape:
        student_model_outputs = student_model(inputs, training=True)
        teacher_model_outputs = teacher_model(inputs, training=False)
        loss = loss_fn(teacher_model_outputs, student_model_outputs)

      grads = tape.gradient(loss, training_vars)
      optimizer.apply_gradients(zip(grads, training_vars))
      train_loss_metric.update_state(loss)
      for metric in train_metrics:
        metric.update_state(labels, tf.nn.softmax(student_model_outputs[-1]))

    @tf.function
    def train_steps(iterator, steps):
      if not isinstance(steps, tf.Tensor):
        raise ValueError('steps should be an Tensor. Python object may cause '
                         'retracing.')

      for _ in tf.range(steps):
        strategy.experimental_run_v2(_replicated_step, args=(next(iterator),))

    def train_single_step(iterator):
      strategy.experimental_run_v2(_replicated_step, args=(next(iterator),))

    def test_step(iterator):

      def _test_step_fn(inputs):

        inputs, labels = inputs
        student_model_outputs = student_model(inputs, training=False)
        for metric in eval_metrics:
          metric.update_state(labels, student_model_outputs[-1])

      strategy.experimental_run_v2(_test_step_fn, args=(next(iterator),))

    train_single_step = tf.function(train_single_step)
    test_step = tf.function(test_step)

    def _run_evaluation(current_training_step, test_iterator):
      for _ in range(eval_steps):
        test_step(test_iterator)

      with eval_summary_writer.as_default():
        for metric in eval_metrics + student_model.metrics:
          metric_value = _float_metric_value(metric)
          logging.info('Step: [%d] Validation %s = %f', current_training_step,
                       metric.name, metric_value)
          tf.summary.scalar(
              metric.name, metric_value, step=current_training_step)
        eval_summary_writer.flush()

    checkpoint = tf.train.Checkpoint(model=student_model, optimizer=optimizer)
    latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)
    if latest_checkpoint_file:
      logging.info(
          'Checkpoint file %s found and restoring from '
          'checkpoint', latest_checkpoint_file)
      checkpoint.restore(latest_checkpoint_file)
      logging.info('Loading from checkpoint file completed')

    current_step = optimizer.iterations.numpy()
    checkpoint_name = 'ctl_step_{step}.ckpt'

    while current_step < total_training_steps:
      train_loss_metric.reset_states()
      for metric in train_metrics + student_model.metrics:
        metric.reset_states()

      steps = steps_to_run(current_step, steps_per_epoch, steps_per_loop)

      if steps == 1:
        train_single_step(train_iterator)
      else:
        train_steps(train_iterator,
                    tf.convert_to_tensor(steps, dtype=tf.int32))
      current_step += steps

      train_loss = _float_metric_value(train_loss_metric)
      training_status = 'Train Step: %d/%d  / loss = %s' % (
          current_step, total_training_steps, train_loss)

      if train_summary_writer:
        with train_summary_writer.as_default():
          tf.summary.scalar(
              train_loss_metric.name, train_loss, step=current_step)
          for metric in train_metrics + student_model.metrics:
            metric_value = _float_metric_value(metric)
            training_status += '  %s = %f' % (metric.name, metric_value)
            tf.summary.scalar(metric.name, metric_value, step=current_step)
          train_summary_writer.flush()
      logging.info(training_status)

      if current_step % steps_per_epoch == 0:
        if current_step < total_training_steps:
          _save_checkpoint(checkpoint, model_dir,
                           checkpoint_name.format(step=current_step))
        if eval_input_fn:
          logging.info('Running evaluation after step: %s.', current_step)
          _run_evaluation(current_step,
                          _get_input_iterator(eval_input_fn, strategy))
          # Re-initialize evaluation metric.
          for metric in eval_metrics + student_model.metrics:
            metric.reset_states()

    _save_checkpoint(checkpoint, model_dir,
                     checkpoint_name.format(step=current_step))
    if eval_input_fn:
      logging.info('Running final evaluation after training is complete.')
      _run_evaluation(current_step,
                      _get_input_iterator(eval_input_fn, strategy))

    training_summary = {
        'total_training_steps': total_training_steps,
        'train_loss': _float_metric_value(train_loss_metric),
    }
    if eval_metrics:
      # TODO(hongkuny): Cleans up summary reporting in text.
      training_summary['last_train_metrics'] = _float_metric_value(
          train_metrics[0])
      training_summary['eval_metrics'] = _float_metric_value(eval_metrics[0])

    write_txt_summary(training_summary, summary_dir)

    return student_model

def run_finetuning(task_name=None,
                   data_dir=None,
                   output_dir=None,
                   config_path=None,
                   teacher_config_path=None,
                   ckpt_path=None,
                   teacher_ckpt_path=None,
                   batch_size=8,
                   epochs=3,
                   learning_rate=5e-5):

  strategy = tf.distribute.MirroredStrategy()
  is_distillation = bool(teacher_config_path)
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  meta_data_path = os.path.join(data_dir, 'meta_{}.json'.format(task_name))
  with tf.io.gfile.GFile(meta_data_path, 'r') as f:
    meta_data = json.load(f)
  train_data_size = meta_data['train_data_size'] + meta_data['transfer_data_size']
  test_data_size = meta_data['test_data_size']
  num_classes = meta_data['num_classes']
  max_seq_length = meta_data['max_seq_length']

  steps_per_epoch = int(train_data_size / batch_size)
  steps_per_loop = min(100, steps_per_epoch)
  warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)
  eval_steps = int(test_data_size / batch_size)

  def train_input_fn(ctx=None):
    return get_dataset(task_name, data_dir, dataset_type='train',
        max_seq_length=max_seq_length, batch_size=batch_size, add_transfer=True)
  def eval_input_fn(ctx=None):
    return get_dataset(task_name, data_dir, dataset_type='test',
        max_seq_length=max_seq_length, batch_size=batch_size, add_transfer=False)

  config = CustomBertConfig.from_json_file(config_path)
  if is_distillation:
    teacher_config = CustomBertConfig.from_json_file(teacher_config_path)

  def get_model(bert_config, optimizer=True):
    bert_model, core_model = (classifier_model(
        bert_config, num_classes, max_seq_length, distillation=is_distillation))
    if optimizer:
      bert_model.optimizer = optimization.create_optimizer(learning_rate, steps_per_epoch * epochs, warmup_steps)
    return bert_model, core_model

  model_fn = functools.partial(get_model, config, True)
  if is_distillation:
    teacher_model_fn = functools.partial(get_model, teacher_config, False)

  def get_loss_fn(labels, logits):
    labels = tf.squeeze(labels)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(
        tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(
        tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return loss

  def get_distill_loss_fn(teacher_output, student_output):
    teacher_num_layers = len(teacher_output[3])
    student_num_layers = len(student_output[3])
    teacher_idx = 0
    teacher_layer_idxs = []
    for student_idx in range(student_num_layers):
      teacher_idx += int(math.ceil((teacher_num_layers - teacher_idx) / (student_num_layers - student_idx)))
      teacher_layer_idxs.append(teacher_idx - 1)

    teacher_attentions = [elem for idx, elem in enumerate(teacher_output[3]) if idx in teacher_layer_idxs]
    teacher_hiddens = [elem for idx, elem in enumerate(teacher_output[4]) if idx in teacher_layer_idxs]
    embedding_loss = tf.math.reduce_mean(tf.losses.mse(teacher_output[2], student_output[2]))
    attention_losses = [
        tf.math.reduce_sum(tf.math.reduce_mean(tf.losses.mse(teacher_score, student_score), axis=[0, 2]))
            / config.num_attention_heads
        for teacher_score, student_score in zip(teacher_attentions, student_output[3])]
    hidden_losses = [
        tf.math.reduce_mean(tf.losses.mse(teacher_hidden, student_hidden))
        for teacher_hidden, student_hidden in zip(teacher_hiddens, student_output[4])]
    prediction_loss = tf.math.reduce_sum(-tf.nn.softmax(teacher_output[5]) * tf.nn.log_softmax(student_output[5]))
    loss = embedding_loss + sum(attention_losses) + sum(hidden_losses) + prediction_loss
    return loss

  loss_fn = get_distill_loss_fn if is_distillation else get_loss_fn

  def metric_fn():
    return tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy', dtype=tf.float32)

  if is_distillation:
    run_finetuning_distilling_loop(
        strategy=strategy,
        teacher_model_fn=teacher_model_fn,
        student_model_fn=model_fn,
        loss_fn=loss_fn,
        model_dir=output_dir,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        eval_steps=eval_steps,
        steps_per_epoch=steps_per_epoch,
        steps_per_loop=steps_per_loop,
        epochs=epochs,
        teacher_init_checkpoint=teacher_ckpt_path,
        student_init_checkpoint=ckpt_path,
        metric_fn=metric_fn)
  else:
    run_customized_training_loop(
        strategy=strategy,
        model_fn=model_fn,
        loss_fn=loss_fn,
        model_dir=output_dir,
        steps_per_epoch=steps_per_epoch,
        steps_per_loop=steps_per_loop,
        epochs=epochs,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        eval_steps=eval_steps,
        init_checkpoint=ckpt_path,
        metric_fn=metric_fn)

def main(_):
  run_finetuning(
      task_name = FLAGS.task_name,
      data_dir=FLAGS.data_dir,
      output_dir=FLAGS.output_dir,
      config_path=FLAGS.config_path,
      teacher_config_path=FLAGS.teacher_config_path,
      ckpt_path=FLAGS.ckpt_path,
      teacher_ckpt_path=FLAGS.teacher_ckpt_path,
      batch_size=FLAGS.batch_size,
      epochs=FLAGS.epochs,
      learning_rate=FLAGS.learning_rate)

if __name__ == '__main__':
  app.run(main)
