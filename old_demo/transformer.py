import tensorflow as tf
from tensorflow.keras import layers
from official.nlp.bert_modeling import Transformer, EmbeddingLookup
import copy

class TransformerModel(layers.Layer):
  def __init__(self, config, **kwargs):
    super(TransformerModel, self).__init__(**kwargs)
    self.config = copy.deepcopy(config)

  def build(self, input_shapes):
    self.embedding_lookup = EmbeddingLookup(
        vocab_size=self.config['vocab_size'],
        embedding_size=self.config['hidden_size'],
        initializer_range=self.config['initializer_range'])
    self.transformer_layer = Transformer(
        num_hidden_layers=self.config['num_hidden_layers'],
        hidden_size=self.config['hidden_size'],
        num_attention_heads=self.config['num_attention_heads'],
        intermediate_size=self.config['intermediate_size'],
        intermediate_activation=self.config['hidden_act'],
        hidden_dropout_prob=self.config['hidden_dropout_prob'],
        attention_probs_dropout_prob=self.config['attention_probs_dropout_prob'],
        initializer_range=self.config['initializer_range'])
    self.dense_dropout = layers.Dropout(rate=self.config['hidden_dropout_prob'])
    self.dense_layer = layers.Dense(
        self.config['num_labels'],
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.config['initializer_range']))
    super(TransformerModel, self).build(input_shapes)

  def call(self, inputs):
    word_embedding = self.embedding_lookup(inputs)
    transformer_output = self.transformer_layer(word_embedding)
    transformer_output = tf.squeeze(transformer_output[:, 0:1, :], axis=1)
    dense_dropout_output = self.dense_dropout(transformer_output)
    dense_output = self.dense_layer(dense_dropout_output)
    return dense_output
