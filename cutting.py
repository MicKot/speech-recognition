import pandas as pd
import scipy.io.wavfile as siow
import os
import argparse


def main(args):
    wav_list = []
    for file in os.listdir(args.source_folder):
        if file.endswith(".wav"):
            wav_list.append(os.path.join(args.source_folder, file))

    txt_list = []
    for file in os.listdir("data/"):
        if file.endswith(".txt"):
            txt_list.append(os.path.join(args.source_folder, file))

    for x in range(len(wav_list)):
        fs, signal = siow.read(wav_list[x])
        wav_txt = pd.read_table(txt_list[x], header=None, dtype=str)
        starters = (wav_txt[0].str.replace(',','.').astype(float))*fs
        enders = (wav_txt[1].str.replace(',','.').astype(float))*fs

        for y in range(starters.shape[0]):
            temp_signal=signal[int(starters[y]):int(enders[y])]
            if not os.path.exists('cut_data'):
                os.makedirs('cut_data')
            siow.write('cut_data/{}_{}.wav'.format(wav_txt[2][y], x+1),rate=fs, data=temp_signal)


def arg_parser():
    parser = argparse.ArgumentParser(description='Gets data from source_folder: train/test')
    parser.add_argument("source_folder", help="Folder containing wav files")
    return parser


if __name__ == '__main__':
    main(arg_parser().parse_args())
