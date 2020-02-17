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

limit_gpu_memory(4.5 * 1024.0)

_SUMMARY_TXT = 'training_summary.txt'
_MIN_SUMMARY_STEPS = 10

flags.DEFINE_string('data_dir', default=None, help='Directory for input data in tfrecord format')
flags.DEFINE_string('output_dir', default=None, help='Directory where checkpoints will be saved')
flags.DEFINE_string('config_path', default=None, help='Path to config file')
flags.DEFINE_string('teacher_config_path', default=None, help='Path to config file of teacher model')
flags.DEFINE_string('ckpt_path', default=None, help='Path to checkpoint file')
flags.DEFINE_string('teacher_ckpt_path', default=None, help='Path to checkpoint file of teacher model')
flags.DEFINE_integer('max_seq_length', default=128, help='Max sequence length')
flags.DEFINE_integer('batch_size', default=8, help='Batch size')
flags.DEFINE_integer('epochs', default=100, help='Number of training epochs')
flags.DEFINE_integer('steps_per_epoch', default=10000, help='Number of training steps per epoch')
flags.DEFINE_integer('steps_per_loop', default=100, help='Number of training steps per loop iteration')
flags.DEFINE_integer('warmup_steps', default=10000, help='Warmup steps for optimizer')
flags.DEFINE_float('learning_rate', default=5e-5, help='Initial learning rate')

FLAGS = flags.FLAGS

def decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    example[name] = t

  return example


def create_distill_dataset(input_patterns,
                           seq_length,
                           batch_size):
  """Creates input dataset from (tf)records files for pretraining."""
  name_to_features = {
      'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
  }

  dataset = tf.data.Dataset.list_files(input_patterns, shuffle=True)
  dataset = dataset.repeat()

  input_files = []
  for input_pattern in input_patterns:
    input_files.extend(tf.io.gfile.glob(input_pattern))
  dataset = dataset.shuffle(len(input_files))

  # In parallel, create tf record dataset for each train files.
  # cycle_length = 8 means that up to 8 files will be read and deserialized in
  # parallel. You may want to increase this number if you have a large number of
  # CPU cores.
  dataset = dataset.interleave(
      tf.data.TFRecordDataset, cycle_length=8,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  decode_fn = lambda record: decode_record(record, name_to_features)
  dataset = dataset.map(
      decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def _select_data_from_record(record):
    """Filter out features to use for pretraining."""
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids'],
    }

    y = record['segment_ids']

    return (x, y)

  dataset = dataset.map(
      _select_data_from_record,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.shuffle(100)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(1024)
  return dataset


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

def run_pretraining_distilling_loop(
    strategy=None,
    teacher_model_fn=None,
    student_model_fn=None,
    loss_fn=None,
    model_dir=None,
    train_input_fn=None,
    steps_per_epoch=None,
    steps_per_loop=1,
    epochs=1,
    init_checkpoint=None):

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
    teacher_model = teacher_model_fn()
    student_model = student_model_fn()
    if not hasattr(student_model, 'optimizer'):
      raise ValueError('User should set optimizer attribute to student_model.')
    optimizer = student_model.optimizer

    if init_checkpoint:
      logging.info(
          'Checkpoint file %s found and restoring from '
          'initial checkpoint for core model.', init_checkpoint)
      checkpoint = tf.train.Checkpoint(model=teacher_model)
      checkpoint.restore(init_checkpoint).assert_existing_objects_matched()
      logging.info('Loading from checkpoint file completed')

    train_loss_metric = tf.keras.metrics.Mean(
        'training_loss', dtype=tf.float32)

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

      strategy.experimental_run_v2(_test_step_fn, args=(next(iterator),))

    train_single_step = tf.function(train_single_step)
    test_step = tf.function(test_step)

    def _run_evaluation(current_training_step, test_iterator):
      for _ in range(eval_steps):
        test_step(test_iterator)

      with eval_summary_writer.as_default():
        for metric in student_model.metrics:
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
      for metric in student_model.metrics:
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
          for metric in student_model.metrics:
            metric_value = _float_metric_value(metric)
            training_status += '  %s = %f' % (metric.name, metric_value)
            tf.summary.scalar(metric.name, metric_value, step=current_step)
          train_summary_writer.flush()
      logging.info(training_status)

      if current_step % steps_per_epoch == 0:
        if current_step < total_training_steps:
          _save_checkpoint(checkpoint, model_dir,
                           checkpoint_name.format(step=current_step))

    _save_checkpoint(checkpoint, model_dir,
                     checkpoint_name.format(step=current_step))

    training_summary = {
        'total_training_steps': total_training_steps,
        'train_loss': _float_metric_value(train_loss_metric),
    }

    write_txt_summary(training_summary, summary_dir)

    return student_model


def run_pretraining(data_dir=None,
                    output_dir=None,
                    config_path=None,
                    teacher_config_path=None,
                    ckpt_path=None,
                    teacher_ckpt_path=None,
                    max_seq_length=128,
                    batch_size=32,
                    epochs=100,
                    steps_per_epoch=10000,
                    steps_per_loop=100,
                    warmup_steps=10000,
                    learning_rate=5e-5):

  strategy = tf.distribute.MirroredStrategy()
  input_files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
  is_distillation = bool(teacher_config_path)
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  train_input_fn = functools.partial(create_distill_dataset, input_files,
                                     max_seq_length, batch_size)

  config = CustomBertConfig.from_json_file(config_path)
  if is_distillation:
    teacher_config = CustomBertConfig.from_json_file(teacher_config_path)

  def get_model(bert_config, optimizer=True):
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), name='input_word_ids', dtype=tf.int32)
    input_mask     = tf.keras.layers.Input(shape=(max_seq_length,), name='input_mask', dtype=tf.int32)
    input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), name='input_type_ids', dtype=tf.int32)
    bert_model = get_custom_bert_model(
        input_word_ids, input_mask, input_type_ids, bert_config, 'bert')
    if optimizer:
      bert_model.optimizer = optimization.create_optimizer(
          learning_rate, steps_per_epoch * epochs, warmup_steps)
    return bert_model

  model_fn = functools.partial(get_model, config, True)
  if is_distillation:
    teacher_model_fn = functools.partial(get_model, teacher_config, False)

  def get_loss_fn(labels, model_output):
    return 0

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
    loss = embedding_loss + sum(attention_losses) + sum(hidden_losses)
    return loss

  loss_fn = get_distill_loss_fn if is_distillation else get_loss_fn

  if is_distillation:
    trained_model = run_pretraining_distilling_loop(
        strategy=strategy,
        teacher_model_fn=teacher_model_fn,
        student_model_fn=model_fn,
        loss_fn=loss_fn,
        model_dir=output_dir,
        train_input_fn=train_input_fn,
        steps_per_epoch=steps_per_epoch,
        steps_per_loop=steps_per_loop,
        epochs=epochs,
        init_checkpoint=teacher_ckpt_path)

  return trained_student_model


def main(_):
  run_pretraining(
      data_dir=FLAGS.data_dir,
      output_dir=FLAGS.output_dir,
      config_path=FLAGS.config_path,
      teacher_config_path=FLAGS.teacher_config_path,
      ckpt_path=FLAGS.ckpt_path,
      teacher_ckpt_path=FLAGS.teacher_ckpt_path,
      max_seq_length=FLAGS.max_seq_length,
      batch_size=FLAGS.batch_size,
      epochs=FLAGS.epochs,
      steps_per_epoch=FLAGS.steps_per_epoch,
      steps_per_loop=FLAGS.steps_per_loop,
      warmup_steps=FLAGS.warmup_steps,
      learning_rate=FLAGS.learning_rate)

if __name__ == '__main__':
  app.run(main)
