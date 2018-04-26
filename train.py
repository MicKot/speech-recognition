import argparse
import os
import tensorflow as tf
import preprocessing
import batch_utils as bu
import mfcc
import model


def arg_parser():
    parser = argparse.ArgumentParser(description='Gets data from source_folder: train/test')
    parser.add_argument("source_folder", help="Folder containing wav files")
    return parser


def main(args):
    train_signals, labels, fs = preprocessing.read_and_normalize(args.source_folder)
    train_signals = preprocessing.cut_signal(train_signals, 15800)
    mfcc_signals = mfcc.mfcc_features(train_signals, fs, 1104)
    labels = preprocessing.labels_to_onehot(labels)
    signals_ph = tf.placeholder(tf.float32, shape=[None, mfcc_signals.shape[1], mfcc_signals.shape[2]])
    print(signals_ph.shape)


    y = model.model(signals_ph, 13, 0.8)
    print(y.shape)
    logdir = './graphs'
    sess = tf.InteractiveSession()
    labels_tf = tf.placeholder(tf.int32, [None, labels.shape[1]])
    batcher = bu.Batcher(mfcc_signals, labels)


    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=labels_tf))
        tf.summary.scalar('loss', loss)
    with tf.name_scope("optimizer"):
        train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    with tf.name_scope("evaluation"):
        correct_prediction = tf.equal(tf.argmax(y, 0), tf.argmax(labels, 0))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    train_writer = tf.summary.FileWriter(os.path.join(logdir, "train"))
    train_writer.add_graph(sess.graph)
    summary_op = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())


    TRAIN_STEPS = 2000
    for step in range(TRAIN_STEPS):
        mfcc_sigs, lbls = batcher.next_batch(50)
        summary_result, _ = sess.run([summary_op, train],
                                     feed_dict={signals_ph: mfcc_sigs, labels_tf: lbls})

        train_writer.add_summary(summary_result, step)
        train_writer.add_run_metadata(tf.RunMetadata(), 'step%03d' % step)

    print("Accuracy %f" % sess.run(accuracy, feed_dict={labels_tf: labels}))
    train_writer.close()


if __name__ == '__main__':
    main(arg_parser().parse_args())