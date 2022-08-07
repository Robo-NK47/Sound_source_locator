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


def plot_audio(left_channel, right_channel):
    plt.plot(left_channel.reshape(-1), color='blue', label="Left channel")
    plt.plot(right_channel.reshape(-1), color='red', label="Right channel", alpha=0.6)
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
save_directory = r'D:\pythonProject\Sound_locator\final_label_data'
list_of_files = [f for f in listdir(data_directory) if isfile(join(data_directory, f))]

for i, file_name in enumerate(list_of_files):
    print(f'\nFile {i + 1} out of {len(list_of_files)}')
    data = open_file(os.path.join(data_directory, file_name))

    left_audio = np.array(data['left_audio_data'])
    right_audio = np.array(data['right_audio_data'])

    del data['left_audio_data'], data['right_audio_data']

    left_audio_wavelet = wavelet_transform(left_audio)
    right_audio_wavelet = wavelet_transform(right_audio)

    left_audio = clip_data(normalize_around_mean(left_audio))
    right_audio = clip_data(normalize_around_mean(right_audio))

    left_audio_wavelet = clip_data(normalize_around_mean(left_audio_wavelet))
    right_audio_wavelet = clip_data(normalize_around_mean(right_audio_wavelet))

    plot_audio(left_audio, right_audio)

    data['labels'] = torch.tensor(data['labels'])
    data['left_audio_data'] = torch.tensor(left_audio)
    data['right_audio_data'] = torch.tensor(right_audio)
    data['left_audio_wavelet_data'] = torch.tensor(left_audio_wavelet)
    data['right_audio_wavelet_data'] = torch.tensor(right_audio_wavelet)

    file_path = os.path.join(save_directory, file_name)
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
