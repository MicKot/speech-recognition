from python_speech_features import mfcc
import numpy as np


def mfcc_features(signals, fs, nfft_size):
    mfcc_features = []
    for x, signal in enumerate(signals):
        mfcc_features.append(mfcc(signal, winlen=0.02, winstep=0.01, numcep=20, samplerate=fs[x], nfft=nfft_size))
    return np.array(mfcc_features)

