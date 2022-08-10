import torch
from os import listdir
from os.path import isfile, join
import os
import pickle
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import gc


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
list_of_detection_files = sorted([f for f in listdir(detection_path) if isfile(join(detection_path, f))])
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

comparison_list = []
for dec in list_of_detection_files:
    comparison_list.append(dec[13:])

for raw_data_file_name in list_of_raw_data_files:
    if raw_data_file_name not in comparison_list:
        print(f'Detecting videos at: {raw_data_file_name}')
        data = open_file(os.path.join(raw_data_path, raw_data_file_name))
        bitrate = data['audio_data']['bitrate']
        audio = data['audio_data']['audio']

        left_camera_detections = video_to_detections(data['left_video_data'], model)
        right_camera_detections = video_to_detections(data['right_video_data'], model)

        sampling_factor = 1620
        audio_factor = find_common_divider(len(audio), len(left_camera_detections), sampling_factor)

        audio = audio[0:audio_factor]
        audio = audio.reshape(len(left_camera_detections), -1, 2)

        new_left_result = []
        new_right_result = []
        new_audio = []

        for left_result, right_result, _audio in zip(left_camera_detections, right_camera_detections, audio):
            left_result = left_result.loc[left_result['name'] == 'person']
            right_result = right_result.loc[right_result['name'] == 'person']
            if len(left_result.index) > 0 and len(right_result.index) > 0:
                new_left_result.append(left_result)
                new_right_result.append(right_result)
                new_audio.append(_audio)

        to_save = {'audio_data': {'bitrate': bitrate, 'audio': new_audio},
                   'left_video_data': new_left_result,
                   'right_video_data': new_right_result}

        file_path = os.path.join(detection_path, 'Detections - ' + raw_data_file_name)
        with open(file_path, 'wb') as handle:
            pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        del (audio, new_right_result, new_left_result, new_audio, left_camera_detections, right_camera_detections, data,
             to_save, _audio)
        gc.collect()
