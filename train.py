import argparse
import os
import tensorflow as tf
import operations.preprocessing as preprocessing
import operations.mfcc as mfcc
import operations.model as model


def arg_parser():
    parser = argparse.ArgumentParser(description='Gets data from source_folder: train/test')
    parser.add_argument("source_folder", help="Folder containing wav files")
    parser.add_argument("test_folder", help="Folder containing test files")
    return parser


def main(args):
    train_signals, train_labels, train_fs = preprocessing.read_and_normalize(args.source_folder)
    val_signals, val_labels, val_fs = preprocessing.read_and_normalize(args.test_folder)
    max_len = max(max([len(x)/fx for x, fx in zip(train_signals,train_fs)]), max([len(x)/fx for x, fx in zip(val_signals, val_fs)]))
    train_signals = preprocessing.cut_silence(train_signals, train_fs)
    train_signals = preprocessing.extend_to_max(train_signals, train_fs, max_len)

    val_signals = preprocessing.cut_silence(val_signals, val_fs)
    val_signals = preprocessing.extend_to_max(val_signals, val_fs, max_len)

    train_mfcc = mfcc.mfcc_features(train_signals, train_fs, 1104)
    val_mfcc = mfcc.mfcc_features(val_signals, val_fs, 1104)
    train_labels = preprocessing.labels_to_onehot(train_labels)
    val_labels = preprocessing.labels_to_onehot(val_labels)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_mfcc, train_labels)).batch(len(train_mfcc)).repeat().shuffle(buffer_size=len(train_mfcc))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_mfcc, val_labels)).batch(len(val_mfcc)).repeat().shuffle(buffer_size=len(val_mfcc))

    sess = tf.InteractiveSession()

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    mfcc_data, labels = iterator.get_next()

    training_iterator = train_dataset.make_one_shot_iterator()
    validation_iterator = val_dataset.make_one_shot_iterator()

    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())
    keep_prob = tf.placeholder(tf.float64, shape=())
    training = tf.placeholder(tf.bool, shape=())
    logits = model.model(mfcc_data, 13, keep_prob, training)
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        tf.summary.scalar('loss', loss)
    with tf.name_scope("optimizer"):
        train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    with tf.name_scope("evaluation"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())


    TRAIN_STEPS = 1500
    for step in range(TRAIN_STEPS):
        _, acc, loss_ = sess.run([train, accuracy, loss],
                                     feed_dict={handle: training_handle, keep_prob: 0.5, training: True})

        print("\rstep %d" % step, "train_acc = %f\t" % acc, "loss = %f\t" % loss_, "val_acc = %f"
              % sess.run(accuracy, feed_dict={handle: validation_handle, keep_prob: 1, training: False}))


if __name__ == '__main__':
    main(arg_parser().parse_args())