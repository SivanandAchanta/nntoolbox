#!/usr/bin/python3.6

import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf
from scipy import interpolate
from scipy import signal

def griffinlim(spectrogram, n_iter = 100, window = 'hann', n_fft = 2048, hop_length = -1, verbose = False):
    if hop_length == -1:
        hop_length = n_fft // 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
    for i in range(n_iter):
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length = hop_length, window = window)
        rebuilt = librosa.stft(inverse, n_fft = n_fft, hop_length = hop_length, window = window)
        angles = np.exp(1j * np.angle(rebuilt))

        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length = hop_length, window = window)

    return inverse

'''
log_pow_spec = np.loadtxt('../../feats/train/log_mag_spec/SnowWhite_018_007.sp')
print(log_pow_spec.shape)
pow_spec = np.exp(log_pow_spec)
spec = np.sqrt(pow_spec).transpose(1, 0)
y = griffinlim(spec, n_iter = 200, window = np.hanning, n_fft = 1024, hop_length = 200, verbose = True)
print(y.shape)
v_signal = y / np.max(np.abs(y))
sf.write('temp.wav', v_signal, 16000)
'''


