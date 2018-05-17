import argparse
import os
import tensorflow as tf
import operations.preprocessing as preprocessing
import operations.comp_models as models
import operations.batch_utils as batch_utils
import operations.mfcc as mfcc


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

    train_batcher = batch_utils.Batcher(train_mfcc, train_labels)
    val_batcher = batch_utils.Batcher(val_mfcc, val_labels)


    lbls = tf.placeholder(tf.int32, shape=[None, train_labels.shape[1]])
    input = tf.placeholder(tf.float32, shape=[None, 125, 20])
    dropout_prob = tf.placeholder(tf.float32, shape=())

    model_settings = {'dct_coefficient_count': 20, 'spectrogram_length': 125, 'label_count': 13, 'fingerprint_size': 20}
    
    lgits, _ = models.create_single_fc_model(input, model_settings, 1, dropout_prob)
    
    sess = tf.InteractiveSession()
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=lgits, labels=lbls))
        tf.summary.scalar('loss', loss)
    with tf.name_scope("optimizer"):
        train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    with tf.name_scope("evaluation"):
        correct_prediction = tf.equal(tf.argmax(lgits, 1), tf.argmax(lbls, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())


    TRAIN_STEPS = 1500
    for step in range(TRAIN_STEPS):
        next_data, next_labels = train_batcher.next_batch(len(train_mfcc))
        _, acc, loss_ = sess.run([train, accuracy, loss],
                                     feed_dict={lbls: next_labels, input: next_data, dropout_prob: 0.5})

        val_data, val_labels = val_batcher.next_batch(len(val_mfcc))
        print("\rstep %d" % step, "train_acc = %f\t" % acc, "loss = %f\t" % loss_, "val_acc = %f"
              % sess.run(accuracy, feed_dict={lbls: val_labels, input: val_data, dropout_prob: 1}))


if __name__ == '__main__':
    main(arg_parser().parse_args())