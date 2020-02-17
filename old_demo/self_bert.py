import tensorflow as tf
from tensorflow.keras import layers

import copy

from official.modeling import tf_utils
from official.nlp.bert_modeling import Dense3D, EmbeddingLookup, EmbeddingPostprocessor, get_initializer, create_attention_mask_from_input_mask, Dense2DProjection

class SelfAttention(tf.keras.layers.Layer):
  def __init__(self,
               num_attention_heads=12,
               hidden_size=256,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               **kwargs):
    super(SelfAttention, self).__init__(**kwargs)
    self.num_attention_heads = num_attention_heads
    self.hidden_size = hidden_size
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible

  def build(self, unused_input_shapes):
    self.query_key_dense = self._projection_dense_layer("query_key")
    self.value_concat_dense = self._projection_dense_layer("value_concat")
    self.attention_probs_dropout = tf.keras.layers.Dropout(
        rate=self.attention_probs_dropout_prob)
    super(SelfAttention, self).build(unused_input_shapes)

  def __call__(self, input_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
    return super(SelfAttention, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    (input_tensor, attention_mask) = tf_utils.unpack_inputs(inputs)

    query_key_tensor = self.query_key_dense(input_tensor)

    value_concat_tensor = self.value_concat_dense(input_tensor)

    input_shape = tf_utils.get_shape_list(input_tensor)
    input_tensor_heads = tf.expand_dims(input_tensor, axis=3)
    input_tensor_heads = tf.broadcast_to(input_tensor_heads, input_shape + [self.num_attention_heads])

    attention_scores = tf.einsum("BFND,BSDN->BNSF", query_key_tensor, input_tensor_heads)

    if attention_mask is not None:
      attention_mask = tf.expand_dims(attention_mask, axis=[1])
      adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0
      attention_scores += adder

    attention_probs = tf.nn.softmax(attention_scores)
    attention_probs = self.attention_probs_dropout(attention_probs)

    context_tensor = tf.einsum("BNSF,BFND->BSND", attention_probs, value_concat_tensor)
    context_tensor = tf.reduce_sum(context_tensor, axis=2)

    return context_tensor

  def _projection_dense_layer(self, name):
    """A helper to define a projection layer."""
    return Dense3D(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.hidden_size,
        kernel_initializer=get_initializer(self.initializer_range),
        output_projection=False,
        backward_compatible=self.backward_compatible,
        name=name)

class SelfTransformerBlock(tf.keras.layers.Layer):
  def __init__(self,
               hidden_size=256,
               num_attention_heads=12,
               intermediate_size=1024,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(SelfTransformerBlock, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible
    self.float_type = float_type

  def build(self, unused_input_shapes):
    self.attention_layer = SelfAttention(
        num_attention_heads=self.num_attention_heads,
        hidden_size=self.hidden_size,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_range=self.initializer_range,
        backward_compatible=self.backward_compatible,
        name="self_attention")
    self.attention_dropout = tf.keras.layers.Dropout(
        rate=self.hidden_dropout_prob)
    self.attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm", axis=-1, epsilon=1e-12,
            dtype=tf.float32))
    self.intermediate_dense = Dense2DProjection(
        output_size=self.intermediate_size,
        kernel_initializer=get_initializer(self.initializer_range),
        activation=self.intermediate_activation,
        fp32_activation=True,
        name="intermediate")
    self.output_dense = Dense2DProjection(
        output_size=self.hidden_size,
        kernel_initializer=get_initializer(self.initializer_range),
        name="output")
    self.output_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    super(SelfTransformerBlock, self).build(unused_input_shapes)

  def common_layers(self):
    return [
        self.attention_layer, self.attention_output_dense,
        self.attention_dropout, self.attention_layer_norm,
        self.intermediate_dense, self.output_dense, self.output_dropout,
        self.output_layer_norm
    ]

  def __call__(self, input_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
    return super(SelfTransformerBlock, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    (input_tensor, attention_mask) = tf_utils.unpack_inputs(inputs)
    attention_output = self.attention_layer(
        input_tensor=input_tensor,
        attention_mask=attention_mask)
    attention_output = self.attention_dropout(attention_output)
    attention_output = self.attention_layer_norm(input_tensor +
                                                 attention_output)
    if self.float_type == tf.float16:
      attention_output = tf.cast(attention_output, tf.float16)
    intermediate_output = self.intermediate_dense(attention_output)
    if self.float_type == tf.float16:
      intermediate_output = tf.cast(intermediate_output, tf.float16)
    layer_output = self.output_dense(intermediate_output)
    layer_output = self.output_dropout(layer_output)
    layer_output = self.output_layer_norm(layer_output + attention_output)
    if self.float_type == tf.float16:
      layer_output = tf.cast(layer_output, tf.float16)
    return layer_output

class SelfTransformer(tf.keras.layers.Layer):
  def __init__(self,
               num_hidden_layers=12,
               hidden_size=256,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(SelfTransformer, self).__init__(**kwargs)
    self.num_hidden_layers = num_hidden_layers
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible
    self.float_type = float_type

  def build(self, unused_input_shapes):
    self.layers = []
    for i in range(self.num_hidden_layers):
      self.layers.append(
          SelfTransformerBlock(
              hidden_size=self.hidden_size,
              num_attention_heads=self.num_attention_heads,
              intermediate_size=self.intermediate_size,
              intermediate_activation=self.intermediate_activation,
              hidden_dropout_prob=self.hidden_dropout_prob,
              attention_probs_dropout_prob=self.attention_probs_dropout_prob,
              initializer_range=self.initializer_range,
              backward_compatible=self.backward_compatible,
              float_type=self.float_type,
              name=("layer_%d" % i)))
    super(SelfTransformer, self).build(unused_input_shapes)

  def __call__(self, input_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
    return super(SelfTransformer, self).__call__(inputs=inputs, **kwargs)

  def call(self, inputs, return_all_layers=False):
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    input_tensor = unpacked_inputs[0]
    attention_mask = unpacked_inputs[1]
    output_tensor = input_tensor

    all_layer_outputs = []
    for layer in self.layers:
      output_tensor = layer(output_tensor, attention_mask)
      all_layer_outputs.append(output_tensor)

    if return_all_layers:
      return all_layer_outputs

    return all_layer_outputs[-1]

class SelfBertModel(tf.keras.layers.Layer):
  def __init__(self, config, float_type=tf.float32, **kwargs):
    super(SelfBertModel, self).__init__(**kwargs)
    self.config = (
        BertConfig.from_dict(config)
        if isinstance(config, dict) else copy.deepcopy(config))
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.embedding_lookup = EmbeddingLookup(
        vocab_size=self.config.vocab_size,
        embedding_size=self.config.hidden_size,
        initializer_range=self.config.initializer_range,
        dtype=tf.float32,
        name="word_embeddings")
    self.embedding_postprocessor = EmbeddingPostprocessor(
        use_type_embeddings=True,
        token_type_vocab_size=self.config.type_vocab_size,
        use_position_embeddings=True,
        max_position_embeddings=self.config.max_position_embeddings,
        dropout_prob=self.config.hidden_dropout_prob,
        initializer_range=self.config.initializer_range,
        dtype=tf.float32,
        name="embedding_postprocessor")
    self.encoder = SelfTransformer(
        num_hidden_layers=self.config.num_hidden_layers,
        hidden_size=self.config.hidden_size,
        num_attention_heads=self.config.num_attention_heads,
        intermediate_size=self.config.intermediate_size,
        intermediate_activation=self.config.hidden_act,
        hidden_dropout_prob=self.config.hidden_dropout_prob,
        attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
        initializer_range=self.config.initializer_range,
        backward_compatible=self.config.backward_compatible,
        float_type=self.float_type,
        name="encoder")
    self.pooler_transform = tf.keras.layers.Dense(
        units=self.config.hidden_size,
        activation="tanh",
        kernel_initializer=get_initializer(self.config.initializer_range),
        name="pooler_transform")
    super(SelfBertModel, self).build(unused_input_shapes)

  def __call__(self,
               input_word_ids,
               input_mask=None,
               input_type_ids=None,
               **kwargs):
    inputs = tf_utils.pack_inputs([input_word_ids, input_mask, input_type_ids])
    return super(SelfBertModel, self).__call__(inputs, **kwargs)

  def call(self, inputs, mode="bert"):
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    input_word_ids = unpacked_inputs[0]
    input_mask = unpacked_inputs[1]
    input_type_ids = unpacked_inputs[2]

    word_embeddings = self.embedding_lookup(input_word_ids)
    embedding_tensor = self.embedding_postprocessor(
        word_embeddings=word_embeddings, token_type_ids=input_type_ids)
    if self.float_type == tf.float16:
      embedding_tensor = tf.cast(embedding_tensor, tf.float16)
    attention_mask = None
    if input_mask is not None:
      attention_mask = create_attention_mask_from_input_mask(
          input_word_ids, input_mask)

    if mode == "encoder":
      return self.encoder(
          embedding_tensor, attention_mask, return_all_layers=True)

    sequence_output = self.encoder(embedding_tensor, attention_mask)
    first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    pooled_output = self.pooler_transform(first_token_tensor)

    return (pooled_output, sequence_output)

  def get_config(self):
    config = {"config": self.config.to_dict()}
    base_config = super(SelfBertModel, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

def get_self_bert_model(input_word_ids,
                   input_mask,
                   input_type_ids,
                   config=None,
                   name=None,
                   float_type=tf.float32):
  self_bert_model_layer = SelfBertModel(config=config, float_type=float_type, name=name)
  pooled_output, sequence_output = self_bert_model_layer(input_word_ids, input_mask,
                                                         input_type_ids)
  self_bert_model = tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=[pooled_output, sequence_output])
  return self_bert_model

def classifier_self_model(bert_config,
                     float_type,
                     num_labels,
                     max_seq_length,
                     final_layer_initializer=None,
                     hub_module_url=None):
  input_word_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
  self_bert_model = get_self_bert_model(
      input_word_ids,
      input_mask,
      input_type_ids,
      config=bert_config,
      float_type=float_type)
  pooled_output = self_bert_model.outputs[0]

  initializer = tf.keras.initializers.TruncatedNormal(
      stddev=bert_config.initializer_range)

  output = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(
      pooled_output)
  output = tf.keras.layers.Dense(
      num_labels,
      kernel_initializer=initializer,
      name='output',
      dtype=float_type)(
          output)
  return tf.keras.Model(
      inputs={
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids
      },
      outputs=output), self_bert_model
