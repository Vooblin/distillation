import copy
import json
import math
import six
import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.bert_modeling import EmbeddingLookup, EmbeddingPostprocessor, Dense2DProjection
from official.nlp.bert_modeling import get_initializer, create_attention_mask_from_input_mask


class CustomBertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               embedding_size=None,
               hidden_size=768,
               outer_hidden_size=None,
               inner_hidden_size=None,
               teacher_hidden_size=None,
               num_hidden_layers=12,
               num_hidden_groups=None,
               inner_group_num=1,
               num_attention_heads=12,
               size_per_head=64,
               num_ffn_layers=1,
               intermediate_size=3072,
               hidden_act="gelu",
               use_pooler=True,
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               use_type_embeddings=True,
               type_vocab_size=16,
               initializer_range=0.02,
               backward_compatible=True):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      embedding_size: Size of the embedding layer.
        (ALBERT)
      hidden_size: Size of the encoder layers and the pooler layer.
      outer_hidden_size: Size of the outer hidden states of TransformerBlocks.
        (MobileBERT)
      inner_hidden_size: Size of the inner hidden states of TransformerBlocks.
        (MobileBERT)
      teacher_hidden_size: Size of the encoder layers of teacher model.
        (TinyBERT)
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_hidden_groups: Number of groups of Transformer layers with different
        weights. (ALBERT)
      inner_group_num: Number of layers in every hidden group.
        (ALBERT)
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      size_per_head: Size of every transformer head.
      num_ffn_layers: Number of feed-forward layers.
        (MobileBERT)
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      use_pooler: Flag indicating usage of pooler layer.
        (RoBERTa)
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      use_type_embeddings: Flags indicating usage of type embeddings.
        (RoBERTa)
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      backward_compatible: Boolean, whether the variables shape are compatible
        with checkpoints converted from TF 1.x BERT.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.outer_hidden_size = outer_hidden_size if outer_hidden_size else self.hidden_size
    self.inner_hidden_size = inner_hidden_size if inner_hidden_size else self.hidden_size
    self.embedding_size = embedding_size if embedding_size else self.outer_hidden_size
    self.teacher_hidden_size = teacher_hidden_size if teacher_hidden_size else self.outer_hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_hidden_groups = num_hidden_groups if num_hidden_groups else self.num_hidden_layers
    self.inner_group_num = inner_group_num
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.num_ffn_layers = num_ffn_layers
    self.intermediate_size = intermediate_size
    self.hidden_act = hidden_act
    self.use_pooler = use_pooler
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.use_type_embeddings = use_type_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = CustomBertConfig(**json_object)
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def get_custom_bert_model(input_word_ids,
                          input_mask=None,
                          input_type_ids=None,
                          config=None,
                          name=None,
                          float_type=tf.float32):
  """Wraps the core BERT model as a keras.Model."""
  bert_model_layer = CustomBertModel(config=config, float_type=float_type, name=name)
  outputs = bert_model_layer(input_word_ids, input_mask, input_type_ids)
  bert_model = tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=outputs)
  return bert_model


class CustomBertModel(tf.keras.layers.Layer):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_word_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  input_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  pooled_output, sequence_output = modeling.BertModel(config=config)(
    input_word_ids=input_word_ids,
    input_mask=input_mask,
    input_type_ids=input_type_ids)
  ...
  ```
  """

  def __init__(self, config, float_type=tf.float32, **kwargs):
    super(CustomBertModel, self).__init__(**kwargs)
    self.config = (
        CustomBertConfig.from_dict(config)
        if isinstance(config, dict) else copy.deepcopy(config))
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.embedding_lookup = EmbeddingLookup(
        vocab_size=self.config.vocab_size,
        embedding_size=self.config.embedding_size,
        initializer_range=self.config.initializer_range,
        dtype=tf.float32,
        name="word_embeddings")
    self.embedding_postprocessor = EmbeddingPostprocessor(
        use_type_embeddings=self.config.use_type_embeddings,
        token_type_vocab_size=self.config.type_vocab_size,
        use_position_embeddings=True,
        max_position_embeddings=self.config.max_position_embeddings,
        dropout_prob=self.config.hidden_dropout_prob,
        initializer_range=self.config.initializer_range,
        dtype=tf.float32,
        name="embedding_postprocessor")
    if self.config.embedding_size != self.config.outer_hidden_size:
      self.postembedding_dense = Dense2DProjection(
          output_size=self.config.outer_hidden_size,
          kernel_initializer=get_initializer(self.config.initializer_range),
          name="postembedding_dense")
    if self.config.num_hidden_layers == self.config.num_hidden_groups and self.config.inner_group_num == 1:
      self.encoder = CustomTransformer(
          num_hidden_layers=self.config.num_hidden_layers,
          outer_hidden_size=self.config.outer_hidden_size,
          inner_hidden_size=self.config.inner_hidden_size,
          num_attention_heads=self.config.num_attention_heads,
          size_per_head=self.config.size_per_head,
          num_ffn_layers=self.config.num_ffn_layers,
          intermediate_size=self.config.intermediate_size,
          intermediate_activation=self.config.hidden_act,
          hidden_dropout_prob=self.config.hidden_dropout_prob,
          attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
          initializer_range=self.config.initializer_range,
          backward_compatible=self.config.backward_compatible,
          float_type=self.float_type,
          name="encoder")
    else:
      self.encoders = []
      for group_idx in range(self.config.num_hidden_groups):
        self.encoders.append(CustomTransformer(
            num_hidden_layers=self.config.inner_group_num,
            outer_hidden_size=self.config.outer_hidden_size,
            inner_hidden_size=self.config.inner_hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            size_per_head=self.config.size_per_head,
            num_ffn_layers=self.config.num_ffn_layers,
            intermediate_size=self.config.intermediate_size,
            intermediate_activation=self.config.hidden_act,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            initializer_range=self.config.initializer_range,
            backward_compatible=self.config.backward_compatible,
            float_type=self.float_type,
            name="encoder_{}".format(group_idx)))
    if self.config.use_pooler:
      self.pooler_transform = tf.keras.layers.Dense(
          units=self.config.hidden_size,
          activation="tanh",
          kernel_initializer=get_initializer(self.config.initializer_range),
          name="pooler_transform")
    if self.config.teacher_hidden_size != self.config.outer_hidden_size:
      self.embedding_output_dense = Dense2DProjection(
          output_size=self.config.teacher_hidden_size,
          kernel_initializer=get_initializer(self.config.initializer_range),
          name="embedding_output")
      self.hidden_output_dense = Dense2DProjection(
          output_size=self.config.teacher_hidden_size,
          kernel_initializer=get_initializer(self.config.initializer_range),
          name="hidden_output")
    super(CustomBertModel, self).build(unused_input_shapes)

  def __call__(self,
               input_word_ids,
               input_mask=None,
               input_type_ids=None,
               **kwargs):
    inputs = tf_utils.pack_inputs([input_word_ids, input_mask, input_type_ids])
    return super(CustomBertModel, self).__call__(inputs, **kwargs)

  def call(self, inputs, mode="bert"):
    """Implements call() for the layer.

    Args:
      inputs: packed input tensors.
      mode: string, `bert` or `encoder`.
    Returns:
      Output tensor of the last layer for BERT training (mode=`bert`) which
      is a float Tensor of shape [batch_size, seq_length, hidden_size] or
      a list of output tensors for encoder usage (mode=`encoder`).
    """
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    input_word_ids = unpacked_inputs[0]
    input_mask = unpacked_inputs[1]
    input_type_ids = unpacked_inputs[2]

    word_embeddings = self.embedding_lookup(input_word_ids)
    embedding_tensor = self.embedding_postprocessor(
        word_embeddings=word_embeddings, token_type_ids=input_type_ids)
    if self.float_type == tf.float16:
      embedding_tensor = tf.cast(embedding_tensor, tf.float16)
    if self.config.embedding_size != self.config.outer_hidden_size:
      embedding_tensor = self.postembedding_dense(embedding_tensor)
    attention_mask = None
    if input_mask is not None:
      attention_mask = create_attention_mask_from_input_mask(
          input_word_ids, input_mask)

    if self.config.num_hidden_layers == self.config.num_hidden_groups and self.config.inner_group_num == 1:
      hidden_output, attention_output = self.encoder(embedding_tensor, attention_mask)
    else:
      hidden_output = []
      attention_output = []
      encoder_output = embedding_tensor
      for layer_idx in range(self.config.num_hidden_layers):
        group_idx = layer_idx * self.config.num_hidden_groups // self.config.num_hidden_layers
        encoder_output, attention_output = self.encoders[group_idx](encoder_output, attention_mask)
        hidden_output.extend(encoder_output)
        attention_output.extend(attention_output)
        encoder_output = encoder_output[-1]

    if mode == "encoder":
      return hidden_output

    sequence_output = hidden_output[-1]
    pooled_output = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    if self.config.use_pooler:
      pooled_output = self.pooler_transform(pooled_output)

    embedding_output = embedding_tensor
    if self.config.teacher_hidden_size != self.config.outer_hidden_size:
      embedding_output = self.embedding_output_dense(embedding_tensor)
      hidden_output = [self.hidden_output_dense(h) for h in hidden_output]

    return (pooled_output, sequence_output, embedding_output, attention_output, hidden_output)

  def get_config(self):
    config = {"config": self.config.to_dict()}
    base_config = super(CustomBertModel, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def get_custom_mtdnn_model(input_word_ids,
                           input_mask=None,
                           input_type_ids=None,
                           task_idx=None,
                           bert_layer=None,
                           task_list=None,
                           float_type=tf.float32):
  """Wraps the core BERT model as a keras.Model."""
  mtdnn_model_layer = CustomMtdnnModel(bert_layer=bert_layer, task_list=task_list)
  outputs = mtdnn_model_layer(input_word_ids, input_mask, input_type_ids, task_idx)
  mtdnn_model = tf.keras.Model(
      inputs={
          'input_word_ids': input_word_ids,
          'input_mmask': input_mask,
          'input_type_ids': input_type_ids,
          'task_idx': task_idx},
      outputs=outputs)
  return mtdnn_model


class CustomMtdnnModel(tf.keras.layers.Layer):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_word_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  input_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  pooled_output, sequence_output = modeling.BertModel(config=config)(
    input_word_ids=input_word_ids,
    input_mask=input_mask,
    input_type_ids=input_type_ids)
  ...
  ```
  """

  def __init__(self, bert_layer, task_list, **kwargs):
    super(CustomMtdnnModel, self).__init__(**kwargs)
    self.bert_layer = bert_layer
    self.task_list = task_list

  def build(self, unused_input_shapes):
    self.output_dropout = tf.keras.layers.Dropout(
        rate=self.bert_layer.config.hidden_dropout_prob)
    self.output_layers = []
    for idx, task in enumerate(self.task_list):
      self.output_layers.append(tf.keras.layers.Dense(
          units=task["num_classes"],
          activation="softmax",
          kernel_initializer=get_initializer(self.bert_layer.config.initializer_range),
          name="output_layer_{}".format(idx)))
    super(CustomMtdnnModel, self).build(unused_input_shapes)

  def __call__(self,
               input_word_ids,
               input_mask=None,
               input_type_ids=None,
               task_idx=None,
               **kwargs):
    inputs = tf_utils.pack_inputs([input_word_ids, input_mask, input_type_ids, task_idx])
    return super(CustomMtdnnModel, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    input_word_ids = unpacked_inputs[0]
    input_mask = unpacked_inputs[1]
    input_type_ids = unpacked_inputs[2]
    task_idx = unpacked_inputs[3]

    bert_outputs = self.bert_layer(input_word_ids, input_mask, input_type_ids)
    pooled_output = bert_outputs[0]
    mtdnn_output = self.output_dropout(pooled_output)
    branch_fns = {idx: lambda: self.output_layers[idx](mtdnn_output) for idx in range(len(self.task_list))}
    mtdnn_output = tf.switch_case(task_idx[0], branch_fns)
    return mtdnn_output


class CustomAttention(tf.keras.layers.Layer):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with tf.einsum as follows:
    Input_tensor: [BFD]
    Wq, Wk, Wv: [DNH]
    Q:[BFNH] = einsum('BFD,DNH->BFNH', Input_tensor, Wq)
    K:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wk)
    V:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wv)
    attention_scores:[BNFT] = einsum('BTNH,BFNH->BNFT', K, Q) / sqrt(H)
    attention_probs:[BNFT] = softmax(attention_scores)
    context_layer:[BFNH] = einsum('BNFT,BTNH->BFNH', attention_probs, V)
    Wout:[DNH]
    Output:[BFD] = einsum('BFNH,DNH>BFD', context_layer, Wout)
  """

  def __init__(self,
               num_attention_heads=12,
               size_per_head=64,
               hidden_size=768,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               **kwargs):
    super(CustomAttention, self).__init__(**kwargs)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.hidden_size = hidden_size
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.query_dense = self._projection_dense_layer("query")
    self.key_dense = self._projection_dense_layer("key")
    self.value_dense = self._projection_dense_layer("value")
    self.attention_probs_dropout = tf.keras.layers.Dropout(
        rate=self.attention_probs_dropout_prob)
    super(CustomAttention, self).build(unused_input_shapes)

  def reshape_to_matrix(self, input_tensor):
    """Reshape N > 2 rank tensor to rank 2 tensor for performance."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
      raise ValueError("Input tensor must have at least rank 2."
                       "Shape = %s" % (input_tensor.shape))
    if ndims == 2:
      return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor

  def __call__(self, from_tensor, to_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([from_tensor, to_tensor, attention_mask])
    return super(CustomAttention, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    (from_tensor, to_tensor, attention_mask) = tf_utils.unpack_inputs(inputs)

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query_tensor` = [B, F, N ,H]
    query_tensor = self.query_dense(from_tensor)

    # `key_tensor` = [B, T, N, H]
    key_tensor = self.key_dense(to_tensor)

    # `value_tensor` = [B, T, N, H]
    value_tensor = self.value_dense(to_tensor)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_tensor, query_tensor)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self.size_per_head)))

    if attention_mask is not None:
      # `attention_mask` = [B, 1, F, T]
      attention_mask = tf.expand_dims(attention_mask, axis=[1])

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.attention_probs_dropout(attention_probs)

    # `context_layer` = [B, F, N, H]
    context_tensor = tf.einsum("BNFT,BTNH->BFNH", attention_probs, value_tensor)

    return context_tensor, attention_scores

  def _projection_dense_layer(self, name):
    """A helper to define a projection layer."""
    return CustomDense3D(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.size_per_head,
        hidden_size=self.hidden_size,
        kernel_initializer=get_initializer(self.initializer_range),
        output_projection=False,
        backward_compatible=self.backward_compatible,
        name=name)


class CustomDense3D(tf.keras.layers.Layer):
  """A Dense Layer using 3D kernel with tf.einsum implementation.

  Attributes:
    num_attention_heads: An integer, number of attention heads for each
      multihead attention layer.
    size_per_head: An integer, hidden size per attention head.
    hidden_size: An integer, dimension of the hidden layer.
    kernel_initializer: An initializer for the kernel weight.
    bias_initializer: An initializer for the bias.
    activation: An activation function to use. If nothing is specified, no
      activation is applied.
    use_bias: A bool, whether the layer uses a bias.
    output_projection: A bool, whether the Dense3D layer is used for output
      linear projection.
    backward_compatible: A bool, whether the variables shape are compatible
      with checkpoints converted from TF 1.x.
  """

  def __init__(self,
               num_attention_heads=12,
               size_per_head=72,
               hidden_size=864,
               kernel_initializer=None,
               bias_initializer="zeros",
               activation=None,
               use_bias=True,
               output_projection=False,
               backward_compatible=False,
               **kwargs):
    """Inits Dense3D."""
    super(CustomDense3D, self).__init__(**kwargs)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.hidden_size = hidden_size
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.activation = activation
    self.use_bias = use_bias
    self.output_projection = output_projection
    self.backward_compatible = backward_compatible

  @property
  def compatible_kernel_shape(self):
    if self.output_projection:
      return [self.num_attention_heads * self.size_per_head, self.hidden_size]
    return [self.last_dim, self.num_attention_heads * self.size_per_head]

  @property
  def compatible_bias_shape(self):
    if self.output_projection:
      return [self.hidden_size]
    return [self.num_attention_heads * self.size_per_head]

  @property
  def kernel_shape(self):
    if self.output_projection:
      return [self.num_attention_heads, self.size_per_head, self.hidden_size]
    return [self.last_dim, self.num_attention_heads, self.size_per_head]

  @property
  def bias_shape(self):
    if self.output_projection:
      return [self.hidden_size]
    return [self.num_attention_heads, self.size_per_head]

  def build(self, input_shape):
    """Implements build() for the layer."""
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError("Unable to build `Dense3D` layer with non-floating "
                      "point (and non-complex) dtype %s" % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError("The last dimension of the inputs to `Dense3D` "
                       "should be defined. Found `None`.")
    self.last_dim = tf.compat.dimension_value(input_shape[-1])
    self.input_spec = tf.keras.layers.InputSpec(
        min_ndim=3, axes={-1: self.last_dim})
    # Determines variable shapes.
    if self.backward_compatible:
      kernel_shape = self.compatible_kernel_shape
      bias_shape = self.compatible_bias_shape
    else:
      kernel_shape = self.kernel_shape
      bias_shape = self.bias_shape

    self.kernel = self.add_weight(
        "kernel",
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          "bias",
          shape=bias_shape,
          initializer=self.bias_initializer,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    super(CustomDense3D, self).build(input_shape)

  def call(self, inputs):
    """Implements ``call()`` for Dense3D.

    Args:
      inputs: A float tensor of shape [batch_size, sequence_length, hidden_size]
        when output_projection is False, otherwise a float tensor of shape
        [batch_size, sequence_length, num_heads, dim_per_head].

    Returns:
      The projected tensor with shape [batch_size, sequence_length, num_heads,
        dim_per_head] when output_projection is False, otherwise [batch_size,
        sequence_length, hidden_size].
    """
    if self.backward_compatible:
      kernel = tf.keras.backend.reshape(self.kernel, self.kernel_shape)
      bias = (tf.keras.backend.reshape(self.bias, self.bias_shape)
              if self.use_bias else None)
    else:
      kernel = self.kernel
      bias = self.bias

    if self.output_projection:
      ret = tf.einsum("abcd,cde->abe", inputs, kernel)
    else:
      ret = tf.einsum("abc,cde->abde", inputs, kernel)
    if self.use_bias:
      ret += bias
    if self.activation is not None:
      return self.activation(ret)
    return ret

class CustomTransformerBlock(tf.keras.layers.Layer):
  """Single transformer layer.

  It has two sub-layers. The first is a multi-head self-attention mechanism, and
  the second is a positionwise fully connected feed-forward network.
  """

  def __init__(self,
               outer_hidden_size=768,
               inner_hidden_size=768,
               num_attention_heads=12,
               size_per_head=64,
               num_ffn_layers=1,
               intermediate_size=3072,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(CustomTransformerBlock, self).__init__(**kwargs)
    self.outer_hidden_size = outer_hidden_size
    self.inner_hidden_size = inner_hidden_size
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.num_ffn_layers = num_ffn_layers
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    if self.outer_hidden_size != self.inner_hidden_size:
      self.input_dense = Dense2DProjection(
          output_size=self.inner_hidden_size,
          kernel_initializer=get_initializer(self.initializer_range),
          name="input_dense")
    self.attention_layer = CustomAttention(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.size_per_head,
        hidden_size=self.outer_hidden_size,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_range=self.initializer_range,
        backward_compatible=self.backward_compatible,
        name="self_attention")
    self.attention_output_dense = CustomDense3D(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.size_per_head,
        hidden_size=self.inner_hidden_size,
        kernel_initializer=get_initializer(self.initializer_range),
        output_projection=True,
        backward_compatible=self.backward_compatible,
        name="self_attention_output")
    self.attention_dropout = tf.keras.layers.Dropout(
        rate=self.hidden_dropout_prob)
    self.attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm", axis=-1, epsilon=1e-12,
            # We do layer norm in float32 for numeric stability.
            dtype=tf.float32))
    if self.num_ffn_layers == 1:
      self.intermediate_dense = Dense2DProjection(
          output_size=self.intermediate_size,
          kernel_initializer=get_initializer(self.initializer_range),
          activation=self.intermediate_activation,
          # Uses float32 so that gelu activation is done in float32.
          fp32_activation=True,
          name="intermediate")
      self.output_dense = Dense2DProjection(
          output_size=self.inner_hidden_size,
          kernel_initializer=get_initializer(self.initializer_range),
          name="output")
      self.output_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
      self.output_layer_norm = tf.keras.layers.LayerNormalization(
          name="output_layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    else:
      self.intermediate_denses = []
      self.output_denses = []
      self.output_dropouts = []
      self.output_layer_norms = []
      for idx in range(self.num_ffn_layers):
        self.intermediate_denses.append(Dense2DProjection(
            output_size=self.intermediate_size,
            kernel_initializer=get_initializer(self.initializer_range),
            activation=self.intermediate_activation,
            # Uses float32 so that gelu activation is done in float32.
            fp32_activation=True,
            name="intermediate_{}".format(idx)))
        self.output_denses.append(Dense2DProjection(
            output_size=self.inner_hidden_size,
            kernel_initializer=get_initializer(self.initializer_range),
            name="output_{}".format(idx)))
        self.output_dropouts.append(tf.keras.layers.Dropout(rate=self.hidden_dropout_prob))
        self.output_layer_norms.append(tf.keras.layers.LayerNormalization(
            name="output_layer_norm_{}".format(idx), axis=-1, epsilon=1e-12, dtype=tf.float32))
    if self.outer_hidden_size != self.inner_hidden_size:
      self.last_dense = Dense2DProjection(
          output_size=self.outer_hidden_size,
          kernel_initializer=get_initializer(self.initializer_range),
          name="last_dense")
      self.last_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
      self.last_layer_norm = tf.keras.layers.LayerNormalization(
          name="last_layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    super(CustomTransformerBlock, self).build(unused_input_shapes)

  def common_layers(self):
    """Explicitly gets all layer objects inside a Transformer encoder block."""
    layers = []
    if self.outer_hidden_size != self.inner_hidden_size:
      layers.append(self.input_dense)
    layers.extend([self.attention_layers, self.attention_output_dense,
        self.attention_dropout, self.attention_layer_norm])
    if self.num_ffn_layers == 1:
      layers.extend([self.intermediate_dense, self.output_dense,
          self.output_dropout, self.output_layer_norm])
    else:
      layers.extend([*self.intermediate_denses, *self.output_denses,
          *self.output_dropouts, *self.output_layer_norms])
    if self.outer_hidden_size != self.inner_hidden_size:
      layers.extend([self.last_dense, self.last_dropoutm, self.last_layer_norm])
    return layers

  def __call__(self, input_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
    return super(CustomTransformerBlock, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    (input_tensor, attention_mask) = tf_utils.unpack_inputs(inputs)
    inner_input_tensor = input_tensor
    if self.outer_hidden_size != self.inner_hidden_size:
      inner_input_tensor = self.input_dense(input_tensor)
    attention_output, attention_scores = self.attention_layer(
        from_tensor=input_tensor,
        to_tensor=input_tensor,
        attention_mask=attention_mask)
    attention_output = self.attention_output_dense(attention_output)
    attention_output = self.attention_dropout(attention_output)
    # Use float32 in keras layer norm and the gelu activation in the
    # intermediate dense layer for numeric stability
    attention_output = self.attention_layer_norm(inner_input_tensor +
                                                 attention_output)
    if self.float_type == tf.float16:
      attention_output = tf.cast(attention_output, tf.float16)
    if self.num_ffn_layers == 1:
      intermediate_output = self.intermediate_dense(attention_output)
      if self.float_type == tf.float16:
        intermediate_output = tf.cast(intermediate_output, tf.float16)
      ffn_output = self.output_dense(intermediate_output)
      ffn_output = self.output_dropout(ffn_output)
      # Use float32 in keras layer norm for numeric stability
      ffn_output = self.output_layer_norm(ffn_output + attention_output)
      if self.float_type == tf.float16:
        ffn_output = tf.cast(ffn_output, tf.float16)
    else:
      ffn_output = attention_output
      for idx in range(self.num_ffn_layers):
        old_ffn_output = ffn_output
        intermediate_output = self.intermediate_denses[idx](ffn_output)
        if self.float_type == tf.float16:
          intermediate_output = tf.cast(intermediate_output, tf.float16)
        ffn_output = self.output_denses[idx](intermediate_output)
        ffn_output = self.output_dropouts[idx](ffn_output)
        # Use float32 in keras layer norm for numeric stability
        ffn_output = self.output_layer_norms[idx](ffn_output + old_ffn_output)
        if self.float_type == tf.float16:
           ffn_output = tf.cast(ffn_output, tf.float16)
    layer_output = ffn_output
    if self.outer_hidden_size != self.inner_hidden_size:
      layer_output = self.last_dense(layer_output)
      layer_output = self.last_dropout(layer_output)
      layer_output = self.last_layer_norm(layer_output + input_tensor)
    return layer_output, attention_scores


class CustomTransformer(tf.keras.layers.Layer):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
  """

  def __init__(self,
               num_hidden_layers=12,
               outer_hidden_size=768,
               inner_hidden_size=768,
               num_attention_heads=12,
               size_per_head=64,
               num_ffn_layers=1,
               intermediate_size=3072,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(CustomTransformer, self).__init__(**kwargs)
    self.num_hidden_layers = num_hidden_layers
    self.outer_hidden_size = outer_hidden_size
    self.inner_hidden_size = inner_hidden_size
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.num_ffn_layers = num_ffn_layers
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.layers = []
    for i in range(self.num_hidden_layers):
      self.layers.append(
          CustomTransformerBlock(
              outer_hidden_size=self.outer_hidden_size,
              inner_hidden_size=self.inner_hidden_size,
              num_attention_heads=self.num_attention_heads,
              size_per_head=self.size_per_head,
              num_ffn_layers=self.num_ffn_layers,
              intermediate_size=self.intermediate_size,
              intermediate_activation=self.intermediate_activation,
              hidden_dropout_prob=self.hidden_dropout_prob,
              attention_probs_dropout_prob=self.attention_probs_dropout_prob,
              initializer_range=self.initializer_range,
              backward_compatible=self.backward_compatible,
              float_type=self.float_type,
              name=("layer_%d" % i)))
    super(CustomTransformer, self).build(unused_input_shapes)

  def __call__(self, input_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
    return super(CustomTransformer, self).__call__(inputs=inputs, **kwargs)

  def call(self, inputs, return_all_layers=False):
    """Implements call() for the layer.

    Args:
      inputs: packed inputs.
      return_all_layers: bool, whether to return outputs of all layers inside
        encoders.
    Returns:
      Output tensor of the last layer or a list of output tensors.
    """
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    input_tensor = unpacked_inputs[0]
    attention_mask = unpacked_inputs[1]
    output_tensor = input_tensor

    hidden_output = []
    attention_output = []
    for layer in self.layers:
      output_tensor, attention_scores = layer(output_tensor, attention_mask)
      hidden_output.append(output_tensor)
      attention_output.append(attention_scores)

    return hidden_output, attention_output
