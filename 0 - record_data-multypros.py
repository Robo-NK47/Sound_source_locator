import multiprocessing as mp
import soundcard as sc
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pickle
import time
import os


left_audio_data = []
right_audio_data = []

left_video_data = []
right_video_data = []


def open_file(_file_path):
    with open(_file_path, 'rb') as handle:
        return pickle.load(handle)


def save_object(_path, _data):
    with open(_path, 'wb') as handle:
        pickle.dump(_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def record_left_audio(_attributes):
    with _attributes[0].recorder(samplerate=_attributes[1]) as micro:
        while time.time() < _attributes[2] + 10:
            pass
        print(f'GO! (left audio)   {time.time() - _attributes[2]}')
        while time.time() < _attributes[2] + 30:
            left_audio_data.append(micro.record(numframes=_attributes[3]))
    print(f'Done. (left audio)   {time.time() - _attributes[2]}')
    path = r'D:\pythonProject\Sound_locator\temp'
    _file_name = 'left_audio_data.pkl'
    path = os.path.join(path, _file_name)
    save_object(path, left_audio_data)


def record_right_audio(_attributes):
    with _attributes[0].recorder(samplerate=_attributes[1]) as micro:
        while time.time() < _attributes[2] + 10:
            pass
        print(f'GO! (right audio)   {time.time() - _attributes[2]}')
        while time.time() < _attributes[2] + 30:
            right_audio_data.append(micro.record(numframes=_attributes[3]))
    print(f'Done. (right audio)   {time.time() - _attributes[2]}')
    path = r'D:\pythonProject\Sound_locator\temp'
    _file_name = 'right_audio_data.pkl'
    path = os.path.join(path, _file_name)
    save_object(path, right_audio_data)


def get_left_video(_attributes):
    left_camera = cv.VideoCapture(0)

    if not left_camera.isOpened():
        print("Cannot open cameras")
        exit()
    while time.time() < _attributes + 10:
        pass
    print(f'GO! (left video)   {time.time() - _attributes}')
    while time.time() < _attributes + 30:
        _, left_frame = left_camera.read()

        while left_frame is None:
            _, left_frame = left_camera.read()

        left_video_data.append(left_frame)
    print(f'Done. (left video)   {time.time() - _attributes}')
    path = r'D:\pythonProject\Sound_locator\temp'
    _file_name = 'left_video_data.pkl'
    path = os.path.join(path, _file_name)
    save_object(path, left_video_data)


def get_right_video(_attributes):
    right_camera = cv.VideoCapture(1)

    if not right_camera.isOpened():
        print("Cannot open cameras")
        exit()
    while time.time() < _attributes + 10:
        pass
    print(f'GO! (right video)   {time.time() - _attributes}')
    while time.time() < _attributes + 30:
        _, right_frame = right_camera.read()

        while right_frame is None:
            _, right_frame = right_camera.read()

        right_video_data.append(right_frame)
    print(f'Done. (right video)   {time.time() - _attributes}')
    path = r'D:\pythonProject\Sound_locator\temp'
    _file_name = 'right_video_data.pkl'
    path = os.path.join(path, _file_name)
    save_object(path, right_video_data)


def plot_audio(left_channel, right_channel):
    plt.plot(left_channel, color='blue', label="Left channel")
    plt.plot(right_channel, color='red', label="Right channel", alpha=0.6)
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def play_audio(spkrs, smpl_rt, _audio):
    with spkrs.player(samplerate=smpl_rt) as sp:
        sp.play(_audio)


if __name__ == "__main__":
    ################################################
    # Setup audio
    microphones = sc.all_microphones()
    speakers = sc.all_speakers()
    go_time = time.time()

    for i, skpr in enumerate(speakers):
        print(f'{i} - {skpr.name}')

    speakers = speakers[1]

    for mic in microphones:
        if mic.name == 'Microphone (USB Live camera audio)':
            left_mic = mic

        if mic.name == 'Microphone (2- USB Live camera audio)':
            right_mic = mic

    sample_rate = 48000  # default is 48000
    num_frames = 2 ** 10
    ################################################

    ################################################
    with left_mic.recorder(samplerate=sample_rate) as left_micro, \
            right_mic.recorder(samplerate=sample_rate) as right_micro:
        p1 = mp.Process(target=record_left_audio, args=([left_mic, sample_rate, go_time, num_frames], ))
        p2 = mp.Process(target=record_right_audio, args=([right_mic, sample_rate, go_time, num_frames], ))
        p3 = mp.Process(target=get_left_video, args=(go_time, ))
        p4 = mp.Process(target=get_right_video, args=(go_time, ))

        p1.start()
        p2.start()
        p3.start()
        p4.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
    ################################################
    path = r'D:\pythonProject\Sound_locator\temp'

    left_audio = open_file(os.path.join(path, 'left_audio_data.pkl'))
    left_video = open_file(os.path.join(path, 'left_video_data.pkl'))
    right_audio = open_file(os.path.join(path, 'right_audio_data.pkl'))
    right_video = open_file(os.path.join(path, 'right_video_data.pkl'))

    left_audio_data = np.array(left_audio).reshape(len(left_audio) * num_frames, 2).transpose()[0]
    right_audio_data = np.array(right_audio).reshape(len(right_audio) * num_frames, 2).transpose()[0]

    left_video_data = np.array(left_video)
    right_video_data = np.array(right_video)

    plot_audio(left_audio_data, right_audio_data)
    play_audio(speakers, sample_rate, left_audio_data)
    play_audio(speakers, sample_rate, right_audio_data)

    to_save = {'left_audio_data': left_audio_data,
               'right_audio_data': right_audio_data,
               'left_video_data': left_video_data,
               'right_video_data': right_video_data}

    file_name = f'Raw data - {time.asctime().replace(":", "_")}.pkl'
    file_path = os.path.join(os.getcwd(), 'raw data', file_name)
    save_object(file_path, to_save)
