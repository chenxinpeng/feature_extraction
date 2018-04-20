# 音频特征提取
目前音频可以提取两种特征，一种是传统的 MFCC 特征，另一种是利用 NSynth 音频编码网络提取特征。

## MFCC 特征

### 程序依赖
 - [python_speech_features](https://github.com/jameslyons/python_speech_features)
 - [FFmpeg](https://www.ffmpeg.org/download.html)

### 从视频中抽取音频文件
运行如下脚本，这里以提取 `data/videos/game_of_thrones` 下面视频中的音频为例：
```bash
python prepro_extract_video_wavs.py -video_class game_of_thrones
```

视频的音频文件（`wav`格式）会保存在 `data/wavs/game_of_thrones` 目录下面。

### 从音频文件提取 MFCC 特征
接着，运行如下脚本：
```bash
python prepro_extract_mfcc.py -video_class game_of_thrones
```
MFCC 特征提取时，对于长时间的视频，所需要的计算资源较大，要注意这点。

还有，因为在我们的视频中，从视频文件所提取出的音频文件均有两个通道，记为 C1 与 C2，我们对每个通道分别提取音频的 MFCC 特征。提取特征时，输入的设置参数中，除了输入的 `signal`（音频信号）以及 `samplerate`（采样率）之外，其余的参数为[python_speech_features](https://github.com/jameslyons/python_speech_features) 函数库中默认的参数。如 `winlen`（滑动窗口长度）设置为 25 毫秒，`winstep`（滑动窗口步长）设置为 10 毫秒，更详细的参数信息请参考 [文档](https://github.com/jameslyons/python_speech_features/blob/master/README.rst)。

最后每个通道得到的特征均记为 A1 与 A2，其中 A1 与 A2 的维度是 L x 13，L 等于信号的长度除以采样率。接着我们将其进行拼接，得到整个音频的特征，记为 A，维度为 L x 26。提取出的 MFCC 特征会保存在 `data/audio_feats_mfcc/game_of_thrones` 目录下面，格式为 `numpy` 的 `npy` 格式。

同时，需要注意的是，在音频的每一秒中，我们共可以提取到 100 个 MFCC 特征。于是每一秒的音频特征可以将这一秒内的 MFCC 特征进行拼接或者取平均操作。拼接处理后的每一秒特征为 2600 维，取平均则是 26 维度。


## NSynth 特征
我们借助于 [NSynth](https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth) 对音频进行特征提取。NSynth 一种基于 WaveNet 用于合成声音的自编码器。

### 程序依赖 
 - TensorFlow
 - [librosa](https://github.com/librosa/librosa)
 - [magenta](https://github.com/tensorflow/magenta/tree/master/magenta/models)

### 从音频文件提取 NSynth 特征
先下载 WaveNet 预训练好的模型文件，[Link](https://drive.google.com/file/d/1r7Di_3p-vGXBG5q1LCBbg1tie_Hhzzjd/view?usp=sharing)。将模型文件放在一个指定目录下，这里只要放置的目录与提取脚本中的读取目录一致即可。

于是，在用同样的方式准备好视频的音频文件之后，我们运行如下脚本提取 NSynth 音频特征：
```bash
python prepro_extract_nsynth.py -video_class game_of_thrones
```

### 说明
在上述我们所给的示例代码 `prepro_extract_nsynth.py` 中，对于一段视频 V，我们将其先用 `librosa` 进行读取：
```python
sig, rate = librosa.load(wav_path, sr=16000)
```
`sr` 即为采样率，这里我们取的是 16000，采样率越高，得到的音频信息越全面准确，但相应的 `sig` 信号就会变大，计算量也会随之同比率增加。`wav_path` 为音频文件的位置。在我们的代码中，是将音频信号一段一段的送入 WaveNet 网络中进行特征提取，我们这里取的是每隔 5 秒取一段音频信息：
```python
zip(range(0, sig.shape[0], 5 * rate), range(5 * rate, sig.shape[0], 5 * rate))
```
当将 5 秒长的音频信号输入进 WaveNet 中，得到的是一个 156 x 16 大小的特征矩阵，我们将其拉直得到一个 2496 维的特征向量，这个就作为这 5 秒钟音频信号的特征。

你可以根据自己的需要更改参数配置。
