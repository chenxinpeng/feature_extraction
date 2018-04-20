#! encoding: UTF-8

import os
import ipdb
import glob
import time
import json
import argparse
import numpy as np


def extract_video_wav(video_lists, video_class):
    video_wav_path = 'data/wavs/' + video_class
    if os.path.isdir(video_wav_path) is False:
        os.mkdir(video_wav_path)

    for idx, video_path in enumerate(video_lists):
        time_start = time.time()

        video_wav_save_path = os.path.join(video_wav_path, os.path.basename(video_path).split('.')[0] + '.wav')

        os.system("ffmpeg -i " + video_path + " " + video_wav_save_path)

        print('{}  {}  time cost: {:.3f}'.format(idx, video_path, time.time() - time_start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-video_class', type=str, default='')
    args = parser.parse_args()

    video_lists = glob.glob('data/videos/' + args.video_class + '/*.mp4')

    extract_video_wav(video_lists, args.video_class)