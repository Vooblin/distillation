import tensorflow as tf
from tensorflow.keras import layers
from official.nlp.bert_modeling import Attention, Dense3D, EmbeddingLookup
import copy

class AttentionModel(layers.Layer):
  def __init__(self, config, **kwargs):
    super(AttentionModel, self).__init__(**kwargs)
    self.config = copy.deepcopy(config)

  def build(self, input_shapes):
    self.embedding_lookup = EmbeddingLookup(
        vocab_size=self.config['vocab_size'],
        embedding_size=self.config['hidden_size'])
    self.attention_layers = []
    self.dense3d_layers = []
    self.dense_attention_layers = []
    for _ in range(self.config['num_hidden_layers']):
      self.attention_layers.append(Attention(
          num_attention_heads=self.config['num_attention_heads'],
          size_per_head=self.config['size_per_head'],
          attention_probs_dropout_prob=self.config['attention_probs_dropout_prob']))
      self.dense3d_layers.append(Dense3D(
          num_attention_heads=self.config['num_attention_heads'],
          size_per_head=self.config['size_per_head'],
          output_projection=True))
      self.dense_attention_layers.append(layers.Dense(self.config['hidden_size']))
    self.dense_layer = layers.Dense(self.config['num_labels'])
    super(AttentionModel, self).build(input_shapes)

  def call(self, inputs):
    word_embedding = self.embedding_lookup(inputs)
    intermediate_output = word_embedding
    for i in range(self.config['num_hidden_layers']):
      intermediate_output = self.attention_layers[i](intermediate_output, intermediate_output)
      intermediate_output = self.dense3d_layers[i](intermediate_output)
      intermediate_output = self.dense_attention_layers[i](intermediate_output)
    attention_output = tf.squeeze(intermediate_output[:, 0:1, :], axis=1)
    dense_output = self.dense_layer(attention_output)
    return dense_output
