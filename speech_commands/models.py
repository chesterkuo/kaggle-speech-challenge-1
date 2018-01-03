# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
  """Builds a model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  size specified in model_settings['fingerprint_size'].

  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.

  See the implementations below for the possible model architectures that can be
  requested.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
    Exception: If the architecture type isn't recognized.
  """
  if model_architecture == 'single_fc':
    return create_single_fc_model(fingerprint_input, model_settings,
                                  is_training)
  elif model_architecture == 'conv':
    return create_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'low_latency_conv':
    return create_low_latency_conv_model(fingerprint_input, model_settings,
                                         is_training)
  elif model_architecture == 'low_latency_svdf':
    return create_low_latency_svdf_model(fingerprint_input, model_settings,
                                         is_training, runtime_settings)
#dandbg - start{
  elif model_architecture == 'inception':
    return create_inception_model(fingerprint_input, model_settings,
                                         is_training)
  elif model_architecture == 'resnet':
    return create_resnet_model(fingerprint_input, model_settings,
                                         is_training)
#dandbg - end}
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv, or "low_latency_svdf"')


def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)


def create_single_fc_model(fingerprint_input, model_settings, is_training):
  """Builds a model with a single hidden fully-connected layer.

  This is a very simple model with just one matmul and bias layer. As you'd
  expect, it doesn't produce very accurate results, but it is very fast and
  simple, so it's useful for sanity testing.

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  weights = tf.Variable(
      tf.truncated_normal([fingerprint_size, label_count], stddev=0.001))
  bias = tf.Variable(tf.zeros([label_count]))
  logits = tf.matmul(fingerprint_input, weights) + bias
  if is_training:
    return logits, dropout_prob
  else:
    return logits


def create_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.

  This is roughly the network labeled as 'cnn-trad-fpool3' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces fairly good quality results, but can involve a large number of
  weight parameters and computations. For a cheaper alternative from the same
  paper with slightly less accuracy, see 'low_latency_conv' below.

  During training, dropout nodes are introduced after each relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                             'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_conv_element_count, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_low_latency_conv_model(fingerprint_input, model_settings,
                                  is_training):
  """Builds a convolutional model with low compute requirements.

  This is roughly the network labeled as 'cnn-one-fstride4' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces slightly lower quality results than the 'conv' model, but needs
  fewer weight parameters and computations.

  During training, dropout nodes are introduced after the relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = input_time_size
  first_filter_count = 186
  first_filter_stride_x = 1
  first_filter_stride_y = 4
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
      1, first_filter_stride_y, first_filter_stride_x, 1
  ], 'VALID') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  first_conv_output_width = math.floor(
      (input_frequency_size - first_filter_width + first_filter_stride_x) /
      first_filter_stride_x)
  first_conv_output_height = math.floor(
      (input_time_size - first_filter_height + first_filter_stride_y) /
      first_filter_stride_y)
  first_conv_element_count = int(
      first_conv_output_width * first_conv_output_height * first_filter_count)
  flattened_first_conv = tf.reshape(first_dropout,
                                    [-1, first_conv_element_count])
  first_fc_output_channels = 128
  first_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_conv_element_count, first_fc_output_channels], stddev=0.01))
  first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
  first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 128
  second_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
  second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_fc_output_channels, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_low_latency_svdf_model(fingerprint_input, model_settings,
                                  is_training, runtime_settings):
  """Builds an SVDF model with low compute requirements.

  This is based in the topology presented in the 'Compressing Deep Neural
  Networks using a Rank-Constrained Topology' paper:
  https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
        [SVDF]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This model produces lower recognition accuracy than the 'conv' model above,
  but requires fewer weight parameters and, significantly fewer computations.

  During training, dropout nodes are introduced after the relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    The node is expected to produce a 2D Tensor of shape:
      [batch, model_settings['dct_coefficient_count'] *
              model_settings['spectrogram_length']]
    with the features corresponding to the same time slot arranged contiguously,
    and the oldest slot at index [:, 0], and newest at [:, -1].
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
      ValueError: If the inputs tensor is incorrectly shaped.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']

  # Validation.
  input_shape = fingerprint_input.get_shape()
  if len(input_shape) != 2:
    raise ValueError('Inputs to `SVDF` should have rank == 2.')
  if input_shape[-1].value is None:
    raise ValueError('The last dimension of the inputs to `SVDF` '
                     'should be defined. Found `None`.')
  if input_shape[-1].value % input_frequency_size != 0:
    raise ValueError('Inputs feature dimension %d must be a multiple of '
                     'frame size %d', fingerprint_input.shape[-1].value,
                     input_frequency_size)

  # Set number of units (i.e. nodes) and rank.
  rank = 2
  num_units = 1280
  # Number of filters: pairs of feature and time filters.
  num_filters = rank * num_units
  # Create the runtime memory: [num_filters, batch, input_time_size]
  batch = 1
  memory = tf.Variable(tf.zeros([num_filters, batch, input_time_size]),
                       trainable=False, name='runtime-memory')
  # Determine the number of new frames in the input, such that we only operate
  # on those. For training we do not use the memory, and thus use all frames
  # provided in the input.
  # new_fingerprint_input: [batch, num_new_frames*input_frequency_size]
  if is_training:
    num_new_frames = input_time_size
  else:
    window_stride_ms = int(model_settings['window_stride_samples'] * 1000 /
                           model_settings['sample_rate'])
    num_new_frames = tf.cond(
        tf.equal(tf.count_nonzero(memory), 0),
        lambda: input_time_size,
        lambda: int(runtime_settings['clip_stride_ms'] / window_stride_ms))
  new_fingerprint_input = fingerprint_input[
      :, -num_new_frames*input_frequency_size:]
  # Expand to add input channels dimension.
  new_fingerprint_input = tf.expand_dims(new_fingerprint_input, 2)

  # Create the frequency filters.
  weights_frequency = tf.Variable(
      tf.truncated_normal([input_frequency_size, num_filters], stddev=0.01))
  # Expand to add input channels dimensions.
  # weights_frequency: [input_frequency_size, 1, num_filters]
  weights_frequency = tf.expand_dims(weights_frequency, 1)
  # Convolve the 1D feature filters sliding over the time dimension.
  # activations_time: [batch, num_new_frames, num_filters]
  activations_time = tf.nn.conv1d(
      new_fingerprint_input, weights_frequency, input_frequency_size, 'VALID')
  # Rearrange such that we can perform the batched matmul.
  # activations_time: [num_filters, batch, num_new_frames]
  activations_time = tf.transpose(activations_time, perm=[2, 0, 1])

  # Runtime memory optimization.
  if not is_training:
    # We need to drop the activations corresponding to the oldest frames, and
    # then add those corresponding to the new frames.
    new_memory = memory[:, :, num_new_frames:]
    new_memory = tf.concat([new_memory, activations_time], 2)
    tf.assign(memory, new_memory)
    activations_time = new_memory

  # Create the time filters.
  weights_time = tf.Variable(
      tf.truncated_normal([num_filters, input_time_size], stddev=0.01))
  # Apply the time filter on the outputs of the feature filters.
  # weights_time: [num_filters, input_time_size, 1]
  # outputs: [num_filters, batch, 1]
  weights_time = tf.expand_dims(weights_time, 2)
  outputs = tf.matmul(activations_time, weights_time)
  # Split num_units and rank into separate dimensions (the remaining
  # dimension is the input_shape[0] -i.e. batch size). This also squeezes
  # the last dimension, since it's not used.
  # [num_filters, batch, 1] => [num_units, rank, batch]
  outputs = tf.reshape(outputs, [num_units, rank, -1])
  # Sum the rank outputs per unit => [num_units, batch].
  units_output = tf.reduce_sum(outputs, axis=1)
  # Transpose to shape [batch, num_units]
  units_output = tf.transpose(units_output)

  # Appy bias.
  bias = tf.Variable(tf.zeros([num_units]))
  first_bias = tf.nn.bias_add(units_output, bias)

  # Relu.
  first_relu = tf.nn.relu(first_bias)

  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu

  first_fc_output_channels = 256
  first_fc_weights = tf.Variable(
      tf.truncated_normal([num_units, first_fc_output_channels], stddev=0.01))
  first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
  first_fc = tf.matmul(first_dropout, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 256
  second_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
  second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_fc_output_channels, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc

#dandbg - start{
def add_to_regularization_loss(W):
    tf.add_to_collection("losses", tf.nn.l2_loss(W))

def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))


def create_inception_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.


  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Inception Block]<-(weights/bias)
          v

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  #is_train = tf.get_default_graph().get_tensor_by_name('is_train:0') 
  is_train = is_training

  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  
  #==== 1st conv ====
  first_filter_width = 2
  first_filter_height = 5
  first_filter_count = 128
  stddev = math.sqrt(3.0 / ( 1 + first_filter_count))
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=stddev))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, first_filter_height, first_filter_width, 1],
                            'SAME') + first_bias

  if is_train: 
   first_bn = tf.layers.batch_normalization(first_conv, training = is_train)
  else:
   first_bn = first_conv

  #first_relu = tf.nn.relu(first_conv)
  first_relu = tf.nn.tanh(first_bn)

  first_dropout = first_relu

  #max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  max_pool = first_dropout

  #====inception block 1====
  
  inception_block_N_filter, inception_block_N_filter_count = inception_block(max_pool, first_filter_count, [128, 64, 192, 64, 96, 64], is_train, dropout_prob)

  inception_block_N_1_filter, inception_block_N_1_filter_count = inception_block(inception_block_N_filter, inception_block_N_filter_count, [128, 64, 192, 64, 96, 64], is_train, dropout_prob)

  inception_block_N_2_filter, inception_block_N_2_filter_count = inception_block(inception_block_N_1_filter, inception_block_N_1_filter_count, [128, 64, 192, 64, 96, 64], is_train, dropout_prob)

  inception_block_N_last_filter, inception_block_N_last_filter_count = inception_block(inception_block_N_2_filter, inception_block_N_2_filter_count, [128, 64, 192, 64, 96, 64], is_train, dropout_prob)

  #====2nd conv====
  second_filter_width = 2
  second_filter_height = 5
  second_filter_count = 480
  fc_count = 2048

  stddev = math.sqrt(3.0 / ( inception_block_N_last_filter_count + second_filter_count))
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, inception_block_N_last_filter_count,
              second_filter_count
          ],
          stddev=stddev))

  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(inception_block_N_last_filter, second_weights, [1, second_filter_height, second_filter_width, 1],
                             'SAME') + second_bias
  if is_train:
    second_bn = tf.layers.batch_normalization(second_conv, training = is_train)
  else:
    second_bn = second_conv

  #second_relu = tf.nn.relu(second_conv)
  second_relu = tf.nn.tanh(second_bn)

  second_dropout = second_relu
  second_max_pool = second_dropout
  #second_max_pool = tf.nn.max_pool(second_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


  second_conv_shape = second_max_pool.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(second_max_pool,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']


  stddev = math.sqrt(3.0 / ( second_conv_element_count + fc_count))
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_conv_element_count, fc_count], stddev=stddev))
  final_fc_bias = tf.Variable(tf.zeros([fc_count]))

  #final_fc = tf.nn.relu(tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias)
  final_fc = tf.nn.tanh(tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias)

  if is_training:
      final_fc_drop = tf.nn.dropout(final_fc, 0.5)
  else:
      final_fc_drop = final_fc

#=========
  stddev = math.sqrt(3.0 / ( fc_count + fc_count))
  fc1_weight = tf.Variable(
      tf.truncated_normal(
          [fc_count, fc_count], stddev=stddev))
  fc1_bias = tf.Variable(tf.zeros([fc_count]))
  #fc1_layer = tf.nn.relu(tf.matmul(final_fc_drop, fc1_weight) + fc1_bias)
  fc1_layer = tf.nn.tanh(tf.matmul(final_fc_drop, fc1_weight) + fc1_bias)

  fc2_drop = fc1_layer
#========
  stddev = math.sqrt(3.0 / ( fc_count + label_count))
  fc2_weight = tf.Variable(
      tf.truncated_normal(
          [fc_count, label_count], stddev=stddev))
  fc2_bias = tf.Variable(tf.zeros([label_count]))

  output_layer = tf.matmul(fc2_drop, fc2_weight) + fc2_bias
  #output_layer = tf.matmul(final_fc_drop, fc2_weight) + fc2_bias

  if is_training:
    return output_layer, dropout_prob
  else:
    return output_layer

def inception_block(X, X_filter_count, filters, is_train, dropout_prob):
  """

Args:
  X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
  X_filter_count -- X's filter count
  filters -- python list of integers, defining the number of filters in the inception block

Returns:
  X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
  """
  # Retrieve Filters
  branch_0_1x1_count, branch_1_1x1_count, branch_1_3x3_count, branch_2_1x1_count, branch_2_5x5_count,branch_3_1x1_count = filters


#===============================================
  branch_0_1x1_width = 1
  branch_0_1x1_height = 1
  stddev = math.sqrt(3.0 / ( X_filter_count + branch_0_1x1_count))
  branch_0_1x1_weights = tf.Variable(
      tf.truncated_normal(
          [branch_0_1x1_height, branch_0_1x1_width, X_filter_count, branch_0_1x1_count],
          stddev=stddev))
  branch_0_1x1_bias = tf.Variable(tf.zeros([branch_0_1x1_count]))
  branch_0_1x1_conv = tf.nn.conv2d(X, branch_0_1x1_weights, [1, 1, 1, 1],
                            'SAME') + branch_0_1x1_bias

  if is_train:
    branch_0_1x1_bn = tf.layers.batch_normalization(branch_0_1x1_conv, training = is_train)
  else:
    branch_0_1x1_bn = branch_0_1x1_conv

  #branch_0_1x1_relu = tf.nn.relu(branch_0_1x1_conv)
  branch_0_1x1_relu = tf.nn.tanh(branch_0_1x1_bn)

  branch_0_1x1_dropout = branch_0_1x1_relu

#===============================================

#===============================================
  branch_1_1x1_width = 1
  branch_1_1x1_height = 1
  stddev = math.sqrt(3.0 / ( X_filter_count + branch_1_1x1_count))
  branch_1_1x1_weights = tf.Variable(
      tf.truncated_normal(
          [branch_1_1x1_height, branch_1_1x1_width, X_filter_count, branch_1_1x1_count],
          stddev=stddev))
  branch_1_1x1_bias = tf.Variable(tf.zeros([branch_1_1x1_count]))
  branch_1_1x1_conv = tf.nn.conv2d(X, branch_1_1x1_weights, [1, 1, 1, 1],
                            'SAME') + branch_1_1x1_bias

  if is_train:
    branch_1_1x1_bn = tf.layers.batch_normalization(branch_1_1x1_conv, training = is_train)
  else:
    branch_1_1x1_bn = branch_1_1x1_conv

  #branch_1_1x1_relu = tf.nn.relu(branch_1_1x1_conv)
  branch_1_1x1_relu = tf.nn.tanh(branch_1_1x1_bn)
  branch_1_1x1_dropout = branch_1_1x1_relu

  branch_1_3x3_width = 1
  branch_1_3x3_height = 5
  stddev = math.sqrt(3.0 / ( branch_1_1x1_count + branch_1_3x3_count))
  branch_1_3x3_weights = tf.Variable(
      tf.truncated_normal(
          [branch_1_3x3_height, branch_1_3x3_width, branch_1_1x1_count, branch_1_3x3_count],
          stddev=stddev))
  branch_1_3x3_bias = tf.Variable(tf.zeros([branch_1_3x3_count]))
  branch_1_3x3_conv = tf.nn.conv2d(branch_1_1x1_dropout, branch_1_3x3_weights, [1, 1, 1, 1],
                            'SAME') + branch_1_3x3_bias

  if is_train:
    branch_1_3x3_bn = tf.layers.batch_normalization(branch_1_3x3_conv, training = is_train)
  else:
    branch_1_3x3_bn = branch_1_3x3_conv

  branch_1_3x3_relu = tf.nn.tanh(branch_1_3x3_bn)
  branch_1_3x3_dropout = branch_1_3x3_relu
#===============================================

#===============================================
  branch_2_1x1_width = 1
  branch_2_1x1_height = 1
  stddev = math.sqrt(3.0 / ( X_filter_count + branch_2_1x1_count))
  branch_2_1x1_weights = tf.Variable(
      tf.truncated_normal(
          [branch_2_1x1_height, branch_2_1x1_width, X_filter_count, branch_2_1x1_count],
          stddev=stddev))
  branch_2_1x1_bias = tf.Variable(tf.zeros([branch_2_1x1_count]))
  branch_2_1x1_conv = tf.nn.conv2d(X, branch_2_1x1_weights, [1, 1, 1, 1],
                            'SAME') + branch_2_1x1_bias

  if is_train:
    branch_2_1x1_bn = tf.layers.batch_normalization(branch_2_1x1_conv, training = is_train)
  else:
    branch_2_1x1_bn = branch_2_1x1_conv

  #branch_2_1x1_relu = tf.nn.relu(branch_2_1x1_conv)
  branch_2_1x1_relu = tf.nn.tanh(branch_2_1x1_bn)
  branch_2_1x1_dropout = branch_2_1x1_relu


  branch_2_5x5_width = 2 
  branch_2_5x5_height = 10
  stddev = math.sqrt(3.0 / ( branch_2_1x1_count + branch_2_5x5_count))
  branch_2_5x5_weights = tf.Variable(
      tf.truncated_normal(
          [branch_2_5x5_height, branch_2_5x5_width, branch_2_1x1_count, branch_2_5x5_count],
          stddev=stddev))
  branch_2_5x5_bias = tf.Variable(tf.zeros([branch_2_5x5_count]))
  branch_2_5x5_conv = tf.nn.conv2d(branch_2_1x1_dropout, branch_2_5x5_weights, [1, 1, 1, 1],
                            'SAME') + branch_2_5x5_bias

  if is_train:
    branch_2_5x5_bn = tf.layers.batch_normalization(branch_2_5x5_conv, training = is_train)
  else:
    branch_2_5x5_bn = branch_2_5x5_conv

  #branch_2_5x5_relu = tf.nn.relu(branch_2_5x5_conv)
  branch_2_5x5_relu = tf.nn.tanh(branch_2_5x5_bn)
  branch_2_5x5_dropout = branch_2_5x5_relu

#===============================================

#===============================================

  branch_3_max_pool_3x3 = tf.nn.avg_pool(X, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')
  #branch_3_max_pool_3x3 = tf.nn.max_pool(X, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')

  branch_3_1x1_width = 1
  branch_3_1x1_height = 1
  stddev = math.sqrt(3.0 / ( X_filter_count + branch_3_1x1_count))
  branch_3_1x1_weights = tf.Variable(
      tf.truncated_normal(
          [branch_3_1x1_height, branch_3_1x1_width, X_filter_count, branch_3_1x1_count],
          stddev=stddev))
  branch_3_1x1_bias = tf.Variable(tf.zeros([branch_3_1x1_count]))
  branch_3_1x1_conv = tf.nn.conv2d(branch_3_max_pool_3x3, branch_3_1x1_weights, [1, 1, 1, 1],
                            'SAME') + branch_3_1x1_bias

  if is_train:
    branch_3_1x1_bn = tf.layers.batch_normalization(branch_3_1x1_conv, training = is_train)
  else:
    branch_3_1x1_bn = branch_3_1x1_conv

  #branch_3_1x1_relu = tf.nn.relu(branch_3_1x1_conv)
  branch_3_1x1_relu = tf.nn.tanh(branch_3_1x1_bn)
  branch_3_1x1_dropout = branch_3_1x1_relu
#===============================================

#========concat=================================
  inception_block_output_filter_relu = tf.concat([branch_0_1x1_dropout, branch_1_3x3_dropout, branch_2_5x5_dropout, branch_3_1x1_dropout], 3)
#===============================================

  inception_block_output_filter_count = branch_0_1x1_count + branch_1_3x3_count + branch_2_5x5_count + branch_3_1x1_count

  return inception_block_output_filter_relu, inception_block_output_filter_count

def create_resnet_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.


  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Residual Block]<-(weights/bias)
          v

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

#================= stage 1 ==================
  stage1_filters = [64,64,256]
  stage1_conv_filter, stage1_conv_filter_count = conv_block(max_pool, first_filter_count, stage1_filters, s = 2)
  stage1_identity_filter_1, stage1_identity_filter_1_count = identity_block(stage1_conv_filter, stage1_conv_filter_count, stage1_filters)
  stage1_identity_filter_2, stage1_identity_filter_2_count = identity_block(stage1_identity_filter_1, stage1_identity_filter_1_count, stage1_filters)

#================= stage 2 ==================
  stage2_filters = [128,128,512]
  stage2_conv_filter, stage2_conv_filter_count = conv_block(max_pool, first_filter_count, stage2_filters, s = 2)
  stage2_identity_filter_1, stage2_identity_filter_1_count = identity_block(stage2_conv_filter, stage2_conv_filter_count, stage2_filters)
  stage2_identity_filter_2, stage2_identity_filter_2_count = identity_block(stage2_identity_filter_1, stage2_identity_filter_1_count, stage2_filters)
  stage2_identity_filter_3, stage2_identity_filter_3_count = identity_block(stage2_identity_filter_2, stage2_identity_filter_2_count, stage2_filters)

  final_agv_pool = tf.nn.avg_pool(stage2_identity_filter_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
#===============================================

  final_shape = final_agv_pool.get_shape()
  final_output_width = final_shape[2]
  final_output_height = final_shape[1]
  final_output_filter_count = final_shape[3]
  final_output_element_count = int(
      final_output_width * final_output_height *
      final_output_filter_count)
  flattened_final_output = tf.reshape(final_agv_pool,
                                     [-1, final_output_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [final_output_element_count, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(flattened_final_output, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc

def conv_block(X, X_filter_count, filters, s = 2):
  """

Args:
  X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
  X_filter_count -- X's filter count
  filters -- python list of integers, defining the number of filters in the CONV layers of the main path
  s -- Integer, specifying the stride to be used

Returns:
  X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
  """
  # Retrieve Filters
  F1, F2, F3 = filters

  # Save the input value
  X_shortcut = X

  ##### MAIN PATH #####
  # First component of main path 
  conv_block_1st_filter_width = 1
  conv_block_1st_filter_height = 1
  conv_block_1st_filter_count = F1

  conv_block_1st_filter_weights = tf.Variable(
    tf.truncated_normal(
        [
            conv_block_1st_filter_height, conv_block_1st_filter_width, X_filter_count,
            conv_block_1st_filter_count
        ],
        stddev=0.01))

  conv_block_1st_filter_bias = tf.Variable(tf.zeros([conv_block_1st_filter_count]))
  conv_block_1st_filter_conv = tf.nn.conv2d(X, conv_block_1st_filter_weights, [1, s, s, 1], 'VALID') + conv_block_1st_filter_bias
  conv_block_1st_filter_bn = tf.layers.batch_normalization(conv_block_1st_filter_conv)
  conv_block_1st_filter_relu = tf.nn.relu(conv_block_1st_filter_bn)

  # Second component of main path
  conv_block_2nd_filter_width = 3
  conv_block_2nd_filter_height = 3
  conv_block_2nd_filter_count = F2

  conv_block_2nd_filter_weights = tf.Variable(
    tf.truncated_normal(
        [
            conv_block_2nd_filter_height, conv_block_2nd_filter_width, conv_block_1st_filter_count,
            conv_block_2nd_filter_count
        ],
        stddev=0.01))

  conv_block_2nd_filter_bias = tf.Variable(tf.zeros([conv_block_2nd_filter_count]))
  conv_block_2nd_filter_conv = tf.nn.conv2d(conv_block_1st_filter_relu, conv_block_2nd_filter_weights, [1, 1, 1, 1], 'SAME') + conv_block_2nd_filter_bias
  conv_block_2nd_filter_bn = tf.layers.batch_normalization(conv_block_2nd_filter_conv)
  conv_block_2nd_filter_relu = tf.nn.relu(conv_block_2nd_filter_bn)

  conv_block_3rd_filter_width = 1
  conv_block_3rd_filter_height = 1
  conv_block_3rd_filter_count = F3

  conv_block_3rd_filter_weights = tf.Variable(
    tf.truncated_normal(
        [
            conv_block_3rd_filter_height, conv_block_3rd_filter_width, conv_block_2nd_filter_count,
            conv_block_3rd_filter_count
        ],
        stddev=0.01))

  conv_block_3rd_filter_bias = tf.Variable(tf.zeros([conv_block_3rd_filter_count]))
  conv_block_3rd_filter_conv = tf.nn.conv2d(conv_block_2nd_filter_relu, conv_block_3rd_filter_weights, [1, 1, 1, 1], 'VALID') + conv_block_3rd_filter_bias
  conv_block_3rd_filter_bn = tf.layers.batch_normalization(conv_block_3rd_filter_conv)

  ##### SHORTCUT PATH ####
  conv_block_shortcut_filter_width = 1
  conv_block_shortcut_filter_height = 1
  conv_block_shortcut_filter_count = F3

  conv_block_shortcut_filter_weights = tf.Variable(
    tf.truncated_normal(
        [
            conv_block_shortcut_filter_height, conv_block_shortcut_filter_width, X_filter_count,
            conv_block_shortcut_filter_count
        ],
        stddev=0.01))

  conv_block_shortcut_filter_bias = tf.Variable(tf.zeros([conv_block_shortcut_filter_count]))
  conv_block_shortcut_filter_conv = tf.nn.conv2d(X_shortcut, conv_block_shortcut_filter_weights, [1, s, s, 1], 'VALID') + conv_block_shortcut_filter_bias
  conv_block_shortcut_filter_bn = tf.layers.batch_normalization(conv_block_shortcut_filter_conv)

  # Final step: Add shortcut value to main path, and pass it through a RELU activation
  conv_block_output_filter_add = tf.add(conv_block_3rd_filter_bn, conv_block_shortcut_filter_bn)
  conv_block_output_filter_relu = tf.nn.relu(conv_block_output_filter_add)
  conv_block_output_filter_count = F3

  return conv_block_output_filter_relu, conv_block_output_filter_count


def identity_block(X, X_filter_count, filters):
  """

Args:
  X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
  X_filter_count -- X's filter count
  filters -- python list of integers, defining the number of filters in the CONV layers of the main path

Returns:
  X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
  """
  # Retrieve Filters
  F1, F2, F3 = filters
  
  # Save the input value
  X_shortcut = X

  ##### MAIN PATH #####
  # First component of main path 
  identity_block_1st_filter_width = 1
  identity_block_1st_filter_height = 1
  identity_block_1st_filter_count = F1

  identity_block_1st_filter_weights = tf.Variable(
    tf.truncated_normal(
        [
            identity_block_1st_filter_height, identity_block_1st_filter_width, X_filter_count,
            identity_block_1st_filter_count
        ],
        stddev=0.01))

  identity_block_1st_filter_bias = tf.Variable(tf.zeros([identity_block_1st_filter_count]))
  identity_block_1st_filter_conv = tf.nn.conv2d(X, identity_block_1st_filter_weights, [1, 1, 1, 1], 'VALID') + identity_block_1st_filter_bias
  identity_block_1st_filter_bn = tf.layers.batch_normalization(identity_block_1st_filter_conv)
  identity_block_1st_filter_relu = tf.nn.relu(identity_block_1st_filter_bn)

  # Second component of main path
  identity_block_2nd_filter_width = 3
  identity_block_2nd_filter_height = 3
  identity_block_2nd_filter_count = F2

  identity_block_2nd_filter_weights = tf.Variable(
    tf.truncated_normal(
        [
            identity_block_2nd_filter_height, identity_block_2nd_filter_width, identity_block_1st_filter_count,
            identity_block_2nd_filter_count
        ],
        stddev=0.01))

  identity_block_2nd_filter_bias = tf.Variable(tf.zeros([identity_block_2nd_filter_count]))
  identity_block_2nd_filter_conv = tf.nn.conv2d(identity_block_1st_filter_relu, identity_block_2nd_filter_weights, [1, 1, 1, 1], 'SAME') + identity_block_2nd_filter_bias
  identity_block_2nd_filter_bn = tf.layers.batch_normalization(identity_block_2nd_filter_conv)
  identity_block_2nd_filter_relu = tf.nn.relu(identity_block_2nd_filter_bn)

  identity_block_3rd_filter_width = 1
  identity_block_3rd_filter_height = 1
  identity_block_3rd_filter_count = F3

  identity_block_3rd_filter_weights = tf.Variable(
    tf.truncated_normal(
        [
            identity_block_3rd_filter_height, identity_block_3rd_filter_width, identity_block_2nd_filter_count,
            identity_block_3rd_filter_count
        ],
        stddev=0.01))

  identity_block_3rd_filter_bias = tf.Variable(tf.zeros([identity_block_3rd_filter_count]))
  identity_block_3rd_filter_conv = tf.nn.conv2d(identity_block_2nd_filter_relu, identity_block_3rd_filter_weights, [1, 1, 1, 1], 'VALID') + identity_block_3rd_filter_bias
  identity_block_3rd_filter_bn = tf.layers.batch_normalization(identity_block_3rd_filter_conv)

  # Final step: Add shortcut value to main path, and pass it through a RELU activation
  identity_block_output_filter_add = tf.add(identity_block_3rd_filter_bn, X_shortcut)
  identity_block_output_filter_relu = tf.nn.relu(identity_block_output_filter_add)
  identity_block_output_filter_count = F3

  return identity_block_output_filter_relu, identity_block_output_filter_count
#dandbg - end}
