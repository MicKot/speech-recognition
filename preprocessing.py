import os
import numpy as np
import scipy.io.wavfile as siow
from sklearn.preprocessing import LabelBinarizer

def read_and_normalize(source_folder):
    def normalize(signal):
        signal = signal - np.min(signal)
        signal = signal/(np.max(signal))
        return signal
    files = [os.path.join(source_folder, x) for x in os.listdir(source_folder)]
    signals = []
    labels = []
    for file in files:
        fs, data = siow.read(file)
        if len(data.shape)>1:
            data = normalize(data[:,0])
        else:
            data = normalize(data)
        signals.append(data)
        label = file.split('/')[-1].split('_')[0]
        labels.append(label)

    return signals, labels, fs


def cut_signal(signals, nr_of_probes):
    signals_cut = np.array([np.array(signal[:nr_of_probes]).T for signal in signals])
    return signals_cut


def labels_to_onehot(labels):
    lb = LabelBinarizer()
    lbls = lb.fit_transform(labels)
    return np.array(lbls)