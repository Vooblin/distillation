import tensorflow as tf
from tensorflow.keras import layers
from official.nlp.bert_modeling import TransformerBlock, EmbeddingLookup
import copy

class SharedTransformerModel(layers.Layer):
  def __init__(self, config, **kwargs):
    super(SharedTransformerModel, self).__init__(**kwargs)
    self.config = copy.deepcopy(config)

  def build(self, input_shapes):
    self.embedding_lookup = EmbeddingLookup(
        vocab_size=self.config['vocab_size'],
        embedding_size=self.config['hidden_size'])
    self.transformer_layer = TransformerBlock(
        hidden_size=self.config['hidden_size'],
        num_attention_heads=self.config['num_attention_heads'],
        intermediate_size=self.config['intermediate_size'])
    self.dense_layer = layers.Dense(self.config['num_labels'])
    super(SharedTransformerModel, self).build(input_shapes)

  def call(self, inputs):
    word_embedding = self.embedding_lookup(inputs)
    transformer_output = word_embedding
    for _ in range(self.config['num_hidden_layers']):
      transformer_output = self.transformer_layer(transformer_output)
    transformer_output = tf.squeeze(transformer_output[:, 0:1, :], axis=1)
    dense_output = self.dense_layer(transformer_output)
    return dense_output
