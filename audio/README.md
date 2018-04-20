## 音频特征提取

### 程序依赖
 - Python
 - [python_speech_features](https://github.com/jameslyons/python_speech_features)
 - [FFmpeg](https://www.ffmpeg.org/download.html)

### 从视频中抽取音频文件
运行如下脚本，这里以提取 `data/videos/game_of_thrones` 下面视频中的音频为例：
```bash
python prepro_extract_video_wavs.py -video_class game_of_thrones
```

视频的音频文件会保存在`data/wavs/game_of_thrones`目录下面。

### 从音频文件提取 MFCC 特征
接着，运行如下脚本：
```bash
python prepro_extract_mfcc.py -video_class game_of_thrones
```

注意，在腾讯视频中，视频文件所提取出的音频文件均有两个通道，记为 C1 与 C2，我们对每个通道分别提取音频的 MFCC 特征。提取特征时，输入的设置参数中，除了输入的 `signal`（音频信号）以及 `samplerate`（采样率）之外，其余的参数为[python_speech_features](https://github.com/jameslyons/python_speech_features) 函数库中默认的参数。如 `winlen`（滑动窗口长度）设置为 25 毫秒，`winstep`（滑动窗口步长）设置为 10 毫秒，更详细的参数信息请参考[此文档](https://github.com/jameslyons/python_speech_features/blob/master/README.rst)。

最后每个通道得到的特征均记为 A1 与 A2，其中 A1 与 A2 的维度是 L x 13，L 等于 A1 信号的长度除以采样率。接着我们将其进行拼接，得到整个音频的特征，记为 A，维度为 L x 26。提取出的MFCC特征会保存在 `data/wavs_mfcc_feats/game_of_thrones` 目录下面，格式为 `numpy` 的 `npy` 格式。

同时，需要注意的是，在音频的每一秒中，我们共可以提取到 100 个 MFCC 特征。于是每一秒的音频特征可以将这一秒内的 MFCC 特征进行拼接或者取平均操作。拼接处理后的每一秒特征为 2600 维，取平均则是 26 维度。



