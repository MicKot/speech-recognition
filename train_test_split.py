import os
from sklearn.model_selection import train_test_split
import argparse
import shutil


def arg_parser():
    parser = argparse.ArgumentParser(description='Splits folder into two subdirectories: train and test')
    parser.add_argument("source_folder", help="Folder containing wav files")
    return parser


def move_files(source, dest, files):
    if not os.path.exists(os.path.join(source, dest )):
        os.makedirs(os.path.join(source, dest))
    for file in files:
        shutil.move(file, os.path.join(source, dest))


def main(args):
    files = [os.path.join(args.source_folder, file) for file in os.listdir(args.source_folder)]
    labels = [file.split('/')[-1].split('_')[0] for file in files]
    X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size=0.2)
    move_files(args.source_folder, 'train', X_train)
    move_files(args.source_folder, 'test', X_test)


if __name__ == '__main__':
    main(arg_parser().parse_args())