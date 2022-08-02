import torch
from os import listdir
from os.path import isfile, join
import os
import pickle
import pydub
import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def plot_audio(left_channel, right_channel):
    plt.plot(left_channel, color='blue', label="Left channel")
    plt.plot(right_channel, color='red', label="Right channel", alpha=0.6)
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mygraph.png")


def mp3_to_numpy(_path, normalized=False):
    pydub.AudioSegment.ffmpeg = r'C:\ffmpeg\bin\ffmpeg.exe'
    pydub.AudioSegment.ffprobe = r'C:\ffmpeg\bin\ffprobe.exe'

    a = pydub.AudioSegment.from_mp3(Path(_path))
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2)).transpose()[0].transpose()
    if normalized:
        return np.float32(y) / 2 ** 15
    else:
        return y


def mp4_to_detections(_path, _model):
    cap = cv2.VideoCapture(_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []

    for _ in tqdm(range(frame_count)):
        _, frame = cap.read()
        _results = model(frame)
        results.append(_results.pandas().xyxy[0])

    cap.release()

    return results


def open_file(_file_path):
    with open(_file_path, 'rb') as _handle:
        return pickle.load(_handle)


def get_file_list():
    left_audio_path = os.path.join(os.getcwd(), 'raw data', 'left', 'audio')
    list_of_left_audio_names = sorted([f for f in listdir(left_audio_path) if isfile(join(left_audio_path, f))])

    right_audio_path = os.path.join(os.getcwd(), 'raw data', 'right', 'audio')
    list_of_right_audio_names = sorted([f for f in listdir(right_audio_path) if isfile(join(right_audio_path, f))])

    left_video_path = os.path.join(os.getcwd(), 'raw data', 'left', 'video')
    list_of_left_video_names = sorted([f for f in listdir(left_video_path) if isfile(join(left_video_path, f))])

    right_video_path = os.path.join(os.getcwd(), 'raw data', 'right', 'video')
    list_of_right_video_names = sorted([f for f in listdir(right_video_path) if isfile(join(right_video_path, f))])

    all_file_names = []
    for _left_audio, _right_audio, left_video, right_video in zip(list_of_left_audio_names, list_of_right_audio_names,
                                                                  list_of_left_video_names, list_of_right_video_names):
        left_audio_path = os.path.join(os.getcwd(), 'raw data', 'left', 'audio', _left_audio)
        right_audio_path = os.path.join(os.getcwd(), 'raw data', 'right', 'audio', _right_audio)
        left_video_path = os.path.join(os.getcwd(), 'raw data', 'left', 'video', left_video)
        right_video_path = os.path.join(os.getcwd(), 'raw data', 'right', 'video', right_video)

        dict = {'left_audio_data': left_audio_path, 'right_audio_data': right_audio_path,
                'left_video_data': left_video_path, 'right_video_data': right_video_path}

        all_file_names.append(dict)

    return all_file_names, _left_audio.replace('.mp3', '')


def find_common_divider(num, video_divider, sample_rate):


    while True:
        if num % video_divider == 0 and num % sample_rate == 0:
            return num

        num -= 1


list_of_raw_data_files, file_name = get_file_list()
detection_path = os.path.join(os.getcwd(), 'detection data')
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

for raw_data_file_name in list_of_raw_data_files:
    left_audio_data = mp3_to_numpy(raw_data_file_name['left_audio_data'])
    right_audio_data = mp3_to_numpy(raw_data_file_name['right_audio_data'])

    plot_audio(left_audio_data, right_audio_data)

    left_camera_detections = mp4_to_detections(raw_data_file_name['left_video_data'], model)
    right_camera_detections = mp4_to_detections(raw_data_file_name['right_video_data'], model)

    sampling_factor = 1600
    left_audio_data_factor = find_common_divider(len(left_audio_data), len(left_camera_detections), sampling_factor)
    right_audio_data_factor = find_common_divider(len(right_audio_data), len(right_camera_detections), sampling_factor)

    left_audio_data = left_audio_data[0:left_audio_data_factor]
    right_audio_data = right_audio_data[0:right_audio_data_factor]

    left_audio_data = left_audio_data.reshape(-1, sampling_factor)
    right_audio_data = right_audio_data.reshape(-1, sampling_factor)

    new_left_result = []
    new_right_result = []
    new_left_audio = []
    new_right_audio = []

    for left_result, right_result, left_audio, right_audio in zip(left_camera_detections, right_camera_detections,
                                                                  left_audio_data, right_audio_data):
        left_result = left_result.loc[left_result['name'] == 'person']
        right_result = right_result.loc[right_result['name'] == 'person']
        if len(left_result.index) > 0 and len(right_result.index) > 0:
            new_left_result.append(left_result)
            new_right_result.append(right_result)
            new_left_audio.append(left_audio)
            new_right_audio.append(right_audio)

    to_save = {'left_audio_data': new_left_audio,
               'right_audio_data': new_right_audio,
               'left_video_data': new_left_result,
               'right_video_data': new_right_result}

    file_path = os.path.join(detection_path, 'Detections - ' + file_name + ".pkl")
    with open(file_path, 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
