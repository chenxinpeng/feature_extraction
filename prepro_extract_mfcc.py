#! encoding: UTF-8

import os
import glob
import time
import argparse
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav


def extract_video_wav(video_wavs_list, video_class):
    video_wav_path = 'data/wavs_mfcc_feats/' + video_class
    if os.path.isdir(video_wav_path) is False:
        os.mkdir(video_wav_path)

    for idx, wav_path in enumerate(video_wavs_list):
        time_start = time.time()

        (rate, sig) = wav.read(wav_path)

        sig_c1 = sig[:, 0]
        sig_c2 = sig[:, 1]

        mfcc_feat_c1 = mfcc(sig_c1, rate)
        mfcc_feat_c2 = mfcc(sig_c2, rate)

        mfcc_feat = np.concatenate((mfcc_feat_c1, mfcc_feat_c2), axis=1)

        video_wav_mfcc_feat_save_path = os.path.join(video_wav_path, os.path.basename(wav_path).split('.')[0] + '.npy')

        if os.path.isfile(video_wav_mfcc_feat_save_path) is True:
            continue

        np.save(video_wav_mfcc_feat_save_path, mfcc_feat)

        print('{}  {}  time cost: {:.3f}'.format(idx, wav_path, time.time()-time_start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-video_class', type=str, default='')
    args = parser.parse_args()

    video_wavs_list = glob.glob('data/wavs/' + args.video_class + '/*.wav')
    extract_video_wav(video_wavs_list, args.video_class)
