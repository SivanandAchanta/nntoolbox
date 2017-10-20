#!/usr/bin/python3


'''
Purpose: To extract audio features from wave files 
Extracted audio features include: raw magnitude spectrum, log-magnitude spectrum, mel-spectrum

I/P:
[1] wav directory path

O/P:
[1] Log-Mag spectrum
[2] Mel-Spectrum

Note: These features are extracted to train a seq2seq model (like Tacotron)


Author: Sivanand Achanta

Date V0: 03-09-2017

'''

import argparse
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from python_speech_features import logfbank
import os

def get_logfbank(sig, rate, winlen, shift, nfft, nfilt, opt):

    fb = logfbank(sig, rate, winlen=opt.frsize, winstep=opt.frshift, nfilt=nfilt, nfft=nfft, winfunc=np.hanning)
    return(fb)

def show_magspec(log_sp):

    width = 20
    height = 20
    plt.figure(figsize=(width, height))

    plt.imshow(log_sp.transpose()[-1:0:-1,:])    
    plt.ylabel('frequency')
    plt.xlabel('time')
    plt.show()

def extract_melfeats(opt):

    winlen=int(opt.frsize*opt.fs)
    shift=int(opt.fs*opt.frshift)
    nfft=opt.nfft
    nfilt=opt.nfilt
    
    # read wav file
    for f in os.listdir(opt.wav_dir):
        fname, ext = os.path.splitext(f)
        print('Processing file ' + fname)
        wavfile = os.path.join(opt.wav_dir, f)
        [fs,sig]=wav.read(wavfile)    
        
        # get magnitude spectrum and mel-log filter bank features 
        fb = get_logfbank(sig, fs, winlen, shift, nfft, nfilt, opt)
        # print(fb.shape)
        np.savetxt(opt.out_dir + fname + '.sp', fb)
        # print(sp.shape, fb.shape)

        # show spectrogram
        # log_sp = np.log(sp)
        # show_magspec(sp) 
        
        
if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', required=False, type=str, default='../../../../festival_voices/iiith_uk_blizzard17/wav/', help='../../../festival_voices/iiith_uk_blizzard17/wav/')
    parser.add_argument('--out_dir', required=False, type=str, default='../../feats/log_mag_spec/', help='../feats/afeats/')
    parser.add_argument('--fs', required=False, type=int, default=16000, help='44100')
    parser.add_argument('--frsize', required=False, type=float, default=0.050, help='0.025')
    parser.add_argument('--frshift', required=False, type=float, default=0.0125, help='0.01')
    parser.add_argument('--nfft', required=False, type=int, default=1024, help='1024')
    parser.add_argument('--nfilt', required=False, type=int, default=60, help='26')
    parser.add_argument('--ncep', required=False, type=int, default=13, help='13')
        
    opt = parser.parse_args()
    print(opt)
    
    # prepare the output directories
    try:
        os.makedirs(opt.out_dir)
    except OSError:
        pass

    extract_melfeats(opt)
     

