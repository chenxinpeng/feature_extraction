#! encoding: UTF-8

import os
import glob
import time
import librosa
import argparse
import numpy as np

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from magenta.models.nsynth.wavenet import fastgen


def wavenet_encode(audio):
    # Load the model weights.
    checkpoint_path = './wavenet-ckpt/model.ckpt-200000'

    # Pass the audio through the first half of the autoencoder,
    # to get a list of latent variables that describe the sound.
    # Note that it would be quicker to pass a batch of audio
    # to fastgen.
    encoding = fastgen.encode(audio, checkpoint_path, len(audio))

    # Reshape to a single sound.
    return encoding.reshape([-1, 16])


def extract_video_wav(video_wavs_list):
    for idx, wav_path in enumerate(video_wavs_list):
        time_start = time.time()

        wavnet_feat_save_path = os.path.join(video_wav_path, os.path.basename(wav_path).split('.')[0] + '.npy')
        if os.path.isfile(wavnet_feat_save_path) is True:
            continue

        sig, rate = librosa.load(wav_path, sr=16000)

        wavenet_feat = []
        for start, end in zip(range(0, sig.shape[0], 5 * rate), range(5 * rate, sig.shape[0], 5 * rate)):
            frag_sig = sig[start:end]
            frag_wavenet_feat = wavenet_encode(frag_sig)  # 156 * 16
            wavenet_feat.append(frag_wavenet_feat.reshape([-1]))

        wavenet_feat = np.reshape(wavenet_feat, [-1, 156 * 16])
        np.save(wavnet_feat_save_path, wavenet_feat)

        print('{}  {}  time cost: {:.3f}'.format(idx, wav_path, time.time()-time_start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-start', type=int, default=0)
    parser.add_argument('-end', type=int, default=0)
    args = parser.parse_args()

    video_wav_path = 'data/feats_wavenet/'
    if os.path.isdir(video_wav_path) is False:
        os.mkdir(video_wav_path)

    video_wavs_list = sorted(glob.glob('data/wavs/*.wav'))
    extract_video_wav(video_wavs_list[args.start:args.end])

