import torch
from torch.utils.data import Dataset
import os
import pickle
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm
import torchaudio


def open_file(_file_path):
    with open(_file_path, 'rb') as _handle:
        return pickle.load(_handle)


def loss_function(_label, _model_output):
    return torch.sqrt_(torch.sum(torch.square_(_label - _model_output)))


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.list_of_file_names = [f for f in listdir(self.data_dir) if isfile(join(self.data_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.list_of_file_names)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.list_of_file_names[idx])
        _data = open_file(data_path)
        label = _data['labels']
        del _data['labels']
        if self.transform:
            _data = self.transform(_data)
        return _data, label


train_dataloader = CustomDataset(r'D:\pythonProject\Sound_locator\final_label_data')
for (features, labels) in tqdm(train_dataloader):
    left_audio = features['left_audio_data']
    left_audio_wavelet = features['left_audio_wavelet_data']

    right_audio = features['right_audio_data']
    right_audio_wavelet = features['right_audio_wavelet_data']

    pass
