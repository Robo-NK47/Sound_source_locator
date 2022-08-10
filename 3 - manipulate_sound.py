import os
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import librosa.display as lib
from scipy import signal
from tqdm.auto import tqdm
import torch
from scipy.io.wavfile import write
import torchaudio
import sox


def plot_audio(_audio):
    plt.plot(_audio.transpose()[0].transpose().reshape(-1), color='blue', label="Left channel", alpha=0.6)
    plt.plot(_audio.transpose()[1].transpose().reshape(-1), color='red', label="Right channel", alpha=0.6)
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def open_file(_file_path):
    with open(_file_path, 'rb') as _handle:
        return pickle.load(_handle)


def normalize_around_mean(_data):
    counter = 0
    thresh_hold = 20
    while _data.std() > (1 / thresh_hold) + 0.001:
        _data = (_data / (_data.std() * thresh_hold)) + 0.5
        _data = (_data / _data.mean()) - 0.5
        counter += 1
        # print(f'{counter} - mean: {_data.mean():.3f}, std: {_data.std():.8f}')

    return _data


def clip_data(_data):
    _data[_data > 1] = 1
    _data[_data < 0] = 0
    return _data


def show_freq_domain(_data):
    _data = _data.reshape(_data.shape[0], -1)
    plt.figure(figsize=(10, 5))
    lib.specshow(_data, sr=22050, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()


def wavelet_transform(_data):
    wavelets = []
    for cell in tqdm(_data):
        wavelets.append(signal.cwt(cell, signal.ricker, np.arange(1, 129)))

    return np.array(wavelets).swapaxes(1, 2)


def plot_wavelet(_data, frame):
    plt.imshow(_data[frame], extent=[-1, 1, _data[frame].shape[-1], 1], cmap='PRGn', aspect='auto',
               vmax=abs(_data[frame]).max(), vmin=-abs(_data[frame]).max())
    plt.show()


data_directory = r'D:\pythonProject\Sound_locator\labeled data'
temp_directory = r'D:\pythonProject\Sound_locator\temp'
save_directory = r'D:\pythonProject\Sound_locator\final_label_data'

list_of_files = [f for f in listdir(data_directory) if isfile(join(data_directory, f))]

for i, file_name in enumerate(list_of_files):
    print(f'\nFile {i + 1} out of {len(list_of_files)}')
    data = open_file(os.path.join(data_directory, file_name))

    audio = torch.tensor(np.array(data['audio_data']['audio']))
    bitrate = data['audio_data']['bitrate']

    # temp_path = os.path.join(temp_directory, 'audio file.wav')
    # write(temp_path, bitrate, audio.reshape(-1, 2))
    #
    # audio, bitrate = torchaudio.load(temp_path)
    effects = [["lowpass", "-1", "300"], ["speed", "0.8"], ["rate", f"{bitrate}"], ["reverb", "-w"], ]
    waveform2, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(audio, bitrate, effects)

    left_audio_wavelet = wavelet_transform(audio.transpose()[0].transpose())
    right_audio_wavelet = wavelet_transform(audio.transpose()[1].transpose())

    audio = clip_data(normalize_around_mean(audio))

    left_audio_wavelet = clip_data(normalize_around_mean(left_audio_wavelet))
    right_audio_wavelet = clip_data(normalize_around_mean(right_audio_wavelet))

    plot_audio(audio)

    data['labels'] = torch.tensor(data['labels'])
    data['audio_data'] = torch.tensor(audio)
    data['left_audio_wavelet_data'] = torch.tensor(left_audio_wavelet)
    data['right_audio_wavelet_data'] = torch.tensor(right_audio_wavelet)

    file_path = os.path.join(save_directory, file_name)
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

