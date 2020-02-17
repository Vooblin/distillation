import tensorflow as tf
from tensorflow.keras import layers
from official.nlp.bert_modeling import EmbeddingLookup
import copy

class BiRNNModel(layers.Layer):
  def __init__(self, config, **kwargs):
    super(BiRNNModel, self).__init__(**kwargs)
    self.config = copy.deepcopy(config)

  def build(self, rnn_type, input_shapes):
    self.embedding_lookup = EmbeddingLookup(
        vocab_size=self.config['vocab_size'],
        embedding_size=self.config['hidden_size'])
    if rnn_type == 'lstm':
      cell = layers.LSTMCell
    elif rnn_type == 'gru':
      cell = layers.GRUCell
    else:
      ValueError('RNN type must be "lstm" or "gru"')
    cells = []
    for _ in range(self.config['num_hidden_layers']):
      cells.append(cell(self.config['hidden_size']))
    self.rnn = layers.Bidirectional(layers.RNN(cells))
    self.dense = layers.Dense(self.config['num_labels'])
    super(BiRNNModel, self).build(input_shapes)

  def call(self, inputs):
    word_embedding = self.embedding_lookup(inputs)
    rnn_output = self.rnn(word_embedding)
    dense_output = self.dense(rnn_output)
    return dense_output

class BiLSTMModel(BiRNNModel):
  def __init__(self, config, **kwargs):
    super(BiLSTMModel, self).__init__(config, **kwargs)

  def build(self, input_shapes):
    super(BiLSTMModel, self).build('lstm', input_shapes)

class BiGRUModel(BiRNNModel):
  def __init__(self, config, **kwargs):
    super(BiGRUModel, self).__init__(config, **kwargs)

  def build(self, input_shapes):
    super(BiGRUModel, self).build('gru', input_shapes)
