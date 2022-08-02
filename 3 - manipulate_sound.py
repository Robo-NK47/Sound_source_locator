import os
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display as lib


def plot_audio(channel):
    plt.plot(channel, color='blue')
    plt.xlabel("Samples")
    plt.ylabel("Values")
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
        print(f'{counter} - mean: {_data.mean():.3f}, std: {_data.std():.8f}')

    clipped_data = np.zeros(_data.shape)
    for i, cell in enumerate(_data):
        clipped_cell = []
        for entry in cell:
            if entry > 1:
                clipped_cell.append(1)
            elif entry < 0:
                clipped_cell.append(0)
            else:
                clipped_cell.append(entry)

        clipped_data[i] = np.array(clipped_cell)

    return clipped_data


# def fft(_data):
#     _complex = np.fft.fft(_data)
#     _shape = (1, _complex.shape[0], _complex.shape[1])
#
#     _real = _complex.real.reshape(_shape)
#     _imaginary = _complex.imag.reshape(_shape)
#
#     _complex = np.concatenate([_real, _imaginary], axis=0)
#
#     return _complex


def fft(_data):
    to_return = []
    for cell in _data:
        x = librosa.stft(cell.reshape(-1).astype(np.float64))
        xdb = librosa.amplitude_to_db(abs(x))
        to_return.append(xdb)
    return np.array(to_return)


def show_freq_domain(_data):
    _data = _data.reshape(_data.shape[0], -1)
    plt.figure(figsize=(10, 5))
    lib.specshow(_data, sr=22050, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()


data_directory = r'D:\pythonProject\Sound_locator\labeled data'
list_of_files = [f for f in listdir(data_directory) if isfile(join(data_directory, f))]

for file_name in list_of_files:
    data = open_file(os.path.join(data_directory, file_name))

    left_audio = np.array(data['left_audio_data'])
    right_audio = np.array(data['right_audio_data'])

    del data['left_audio_data'], data['right_audio_data']

    left_audio_fft = fft(left_audio)
    right_audio_fft = fft(right_audio)

    left_audio = normalize_around_mean(left_audio)
    right_audio = normalize_around_mean(right_audio)

    print('a')

