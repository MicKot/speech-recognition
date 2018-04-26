import tensorflow as tf


def model(input, num_classes, keep_prob):
    with tf.name_scope('model'):
        # c1 = tf.layers.conv1d(input, filters=100, kernel_size=5, name='c1')
        # fc1 = tf.layers.dense(c1, units=100, activation=tf.nn.relu, name='fc1')
        # fc2 = tf.layers.dense(fc1, units=100, activation=tf.nn.relu, name='fc2')
        # dropped = tf.nn.dropout(fc2, keep_prob=keep_prob, name='dropped')
        y = tf.layers.dense(input, units=num_classes, name="output")
    return y