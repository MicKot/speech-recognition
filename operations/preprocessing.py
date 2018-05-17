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
    fs = []
    for file in files:
        fssig, data = siow.read(file)
        if len(data.shape)>1:
            data = normalize(data[:,0])
        else:
            data = normalize(data)
        signals.append(data)
        label = file.split('/')[-1].split('_')[0]
        labels.append(label)
        fs.append(fssig)

    return signals, labels, fs


def extend_to_max(signals, fs, max_len):
    extended_signals = []
    for signal, fx in zip(signals, fs):
        zeros = np.zeros(int(np.ceil((max_len - len(signal)/fx)*fx)))
        signal = np.append(signal, zeros, axis=0)
        extended_signals.append(signal)
    return extended_signals


def cut_silence(signals, fs):
    cut_signals = []
    for signal, fs in zip(signals, fs):
        length = len(signal)
        time_res = 1/fs
        nr_probes = int(0.01/time_res)
        mean_sig_energy = np.power(signal, [2])
        energy_windows = [sum(mean_sig_energy[x * nr_probes:(x+1)* nr_probes]) for x in range(int(length/nr_probes))]
        pre = energy_windows > min(energy_windows) + 0.15*(max(energy_windows)-min(energy_windows))
        x = 0
        while pre[x] != True:
            x += 1
        cut_signals.append(signal[x*nr_probes:])
    return cut_signals


def labels_to_onehot(labels):
    lb = LabelBinarizer()
    lbls = lb.fit_transform(labels)
    return np.array(lbls)