# coding=utf-8
"""The main model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import re

import six
import tensorflow as tf


class TextCNNConfig(object):
    """Configuration for `Text CNN Model`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=100,
                 hidden_act="relu",
                 filter_sizes=None,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=128,
                 fc_dims=None,
                 initializer_range=0.02):
        """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of embedding.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        if fc_dims is None:
            fc_dims = [128, 64]
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.filter_sizes = filter_sizes
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.fc_dims = fc_dims
        self.initializer_range = initializer_range


    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = TextCNNConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TextCNNModel(object):
    """CNN model
  Example usage:
  ```python
  config = modeling.XXXConfig(vocab_size=32000, hidden_size=512, ...)
  model = modeling.XXXModel(config=config, is_training=True,input_ids=input_ids)
  ...
  ```
  """

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 num_labels,
                 scope=None):
        """Constructor for BertModel.

    Args:
      config: `Config` instance.
      is_training: bool. rue for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      num_labels:   num_labels
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)

        with tf.variable_scope("text_cnn", scope):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings")

            with tf.variable_scope("encoder"):
                self.pooled_output = text_cnn_layers(
                    input_x=self.embedding_output,
                    embed_size=config.hidden_size,
                    filter_sizes=config.filter_sizes,
                    num_filters=config.num_filters,
                    max_position_embeddings=config.max_position_embeddings,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    is_training=is_training
                )

                # full connect layers
                hidden = self.pooled_output
                for idx, dim in enumerate(config.fc_dims):
                    hidden = tf.layers.dense(hidden, units=dim, activation=tf.nn.relu)
                    hidden = tf.layers.dropout(hidden, rate=config.hidden_dropout_prob, training=is_training)

                # shape [num_of_samples, num_of_labels]
                logits = tf.layers.dense(hidden, num_labels, name="logits")

                self.logits = logits

    def get_logits(self):
        return self.logits

    def get_pooled_output(self):
        return self.pooled_output

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).
        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the output of the embedding layer, after summing the word
          embeddings with the positional embeddings and the token type embeddings,
          then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table


def gelu(input_tensor):
    """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  Args:
    input_tensor: float Tensor to perform activation.

  Returns:
    `input_tensor` with the GELU activation applied.
  """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names


def text_cnn_layers(input_x,
                    embed_size,
                    filter_sizes,
                    num_filters,
                    max_position_embeddings,
                    hidden_dropout_prob,
                    is_training):
    """ main computation graph here
        1.embedding --> 2. cnn_layer*2 --> max_pooling
    """
    # get embedding of words in the sentence
    # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
    sentence_embeddings_expanded = tf.expand_dims(input_x, -1)

    # 2.=====>loop each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("convolution-pooling-%s" % filter_size):
            # 1) CNN->BN->relu
            kernel_size = [filter_size, embed_size]
            # conv
            conv = tf.layers.conv2d(sentence_embeddings_expanded,
                                    filters=num_filters,
                                    kernel_size=kernel_size,
                                    strides=[1, 1],
                                    padding="VALID",
                                    name="conv1")

            biases = tf.get_variable(
                'biases', [num_filters],
                initializer=tf.constant_initializer(0.0))

            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, biases), name="relu")

            # shape:[batch_size, sequence_length - filter_size + 1, 1, num_filters]
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, max_position_embeddings - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID")

            pooled_outputs.append(pooled)

    # concate pooled features
    num_of_filters_total = num_filters * len(filter_sizes)
    # [batch_size, num_total_filters]
    h = tf.concat(pooled_outputs, -1)
    # to embeding size
    h = tf.reshape(h, [-1, num_of_filters_total])
    return h


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings"):
    """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))

    output = tf.nn.embedding_lookup(embedding_table, input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return output, embedding_table


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
