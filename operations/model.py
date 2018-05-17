import tensorflow as tf


def model(input, num_classes, keep_prob, training):
    with tf.name_scope('model'):
        r1 = tf.expand_dims(input, axis=3)
        c1 = tf.layers.conv2d(r1, filters=64, kernel_size=[8, 20], strides= [1, 1], name='c1', padding='same')
        relu = tf.nn.leaky_relu(c1)
        dropped = tf.nn.dropout(relu, keep_prob=keep_prob, name='dropped')
        mp1 = tf.nn.max_pool(c1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        c2 = tf.layers.conv2d(mp1, filters=64, kernel_size=[4, 10], strides= [1, 1], name='c2', padding='same')
        relu = tf.nn.leaky_relu(c2)
        dropped = tf.nn.dropout(relu, keep_prob=keep_prob, name='dropped')
        f1 = tf.layers.flatten(dropped)
        y = tf.layers.dense(f1, units=num_classes, name="output", activation= tf.nn.sigmoid)
    return y