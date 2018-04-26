from python_speech_features import mfcc
import numpy as np


def mfcc_features(signals, fs, nfft_size):
    mfcc_features = np.array([mfcc(signal, samplerate=fs, nfft=nfft_size) for signal in signals])
    return mfcc_features

