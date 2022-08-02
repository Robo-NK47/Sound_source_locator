import os
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np


def open_file(_file_path):
    with open(_file_path, 'rb') as _handle:
        return pickle.load(_handle)


data_directory = r'D:\pythonProject\Sound_locator\labeled data'
list_of_files = [f for f in listdir(data_directory) if isfile(join(data_directory, f))]

for file_name in list_of_files:
    data = open_file(os.path.join(data_directory, file_name))

    left_audio = np.array(data['left_audio_data'])
    right_audio = np.array(data['right_audio_data'])

    del data['left_audio_data'], data['right_audio_data']

