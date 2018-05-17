import tensorflow as tf
import math


def create_single_fc_model(fingerprint_input, model_settings, is_training, dropout_prob):
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
    # fingerprint_size = model_settings['fingerprint_size']
    # label_count = model_settings['label_count']
    # weights = tf.Variable(
    #     tf.truncated_normal([fingerprint_size, label_count], stddev=0.001))
    # bias = tf.Variable(tf.zeros([label_count]))
    # logits = tf.matmul(fingerprint_input, weights) + bias
    logits = tf.layers.dense(fingerprint_input, 120, activation=tf.nn.leaky_relu)
    drop = tf.nn.dropout(logits, dropout_prob)
    logits = tf.layers.flatten(drop)
    logits = tf.layers.dense(logits, 13)
    if is_training:
        return logits, dropout_prob
    else:
        return logits


def create_conv_model(fingerprint_input, model_settings, is_training, dropout_prob):
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
    # if is_training:
    #     dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    # input_time_size = model_settings['spectrogram_length']
    # fingerprint_4d = tf.reshape(fingerprint_input,
    #                             [-1, input_time_size, input_frequency_size, 1])
    fingerprint_4d = tf.expand_dims(fingerprint_input, axis=3)
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
                                  is_training, dropout_prob):
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

    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    # fingerprint_4d = tf.reshape(fingerprint_input,
    #                             [-1, input_time_size, input_frequency_size, 1])
    fingerprint_4d = tf.expand_dims(fingerprint_input, axis=3)
    first_filter_width = 8
    first_filter_height = input_time_size
    first_filter_count = 186
    first_filter_stride_x = 1
    first_filter_stride_y = 1
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