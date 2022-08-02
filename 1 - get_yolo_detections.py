import torch
from os import listdir
from os.path import isfile, join
import os
import pickle
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


def video_to_detections(_data, _model):
    results = []
    for frame in tqdm(_data):
        result = model(frame)
        results.append(result.pandas().xyxy[0])

    return results


def open_file(_file_path):
    with open(_file_path, 'rb') as _handle:
        return pickle.load(_handle)


def find_common_divider(num, video_divider, sample_rate):
    while True:
        if num % video_divider == 0 and num % sample_rate == 0:
            return num

        num -= 1


raw_data_path = os.path.join(os.getcwd(), 'raw data')
list_of_raw_data_files = sorted([f for f in listdir(raw_data_path) if isfile(join(raw_data_path, f))])

detection_path = os.path.join(os.getcwd(), 'detection data')
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

for raw_data_file_name in list_of_raw_data_files:
    data = open_file(os.path.join(raw_data_path, raw_data_file_name))
    left_audio_data = data['left_audio_data']
    right_audio_data = data['right_audio_data']

    plot_audio(left_audio_data, right_audio_data)

    left_camera_detections = video_to_detections(data['left_video_data'], model)
    right_camera_detections = video_to_detections(data['right_video_data'], model)

    sampling_factor = 1620
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

    file_path = os.path.join(detection_path, 'Detections - ' + raw_data_file_name)
    with open(file_path, 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
