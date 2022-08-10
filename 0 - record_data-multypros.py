import multiprocessing as mp
import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pickle
import time
import os
import shutil
import wave
from scipy.io.wavfile import read


def get_audio_devices():
    _audio_devices = pyaudio.PyAudio()
    _devices = []
    for _device in range(_audio_devices.get_device_count()):
        _d = _audio_devices.get_device_info_by_index(_device)
        if 'Microphone' in _d['name'] and 'Live camera audio' in _d['name'] and _d['defaultSampleRate'] == 48000.0:
            #print(_d)
            _devices.append(_d)

    return _devices


def get_audio_streams(_chunk, _pa):
    _audio_devices = get_audio_devices()
    sample_format = pyaudio.paInt16  # 16 bits per sample

    _streams = {}
    for audio_device in _audio_devices:
        _streams[audio_device['name']] = _pa.open(format=sample_format, channels=audio_device['maxInputChannels'],
                                                  rate=int(audio_device['defaultSampleRate']), input=True,
                                                  frames_per_buffer=_chunk, input_device_index=audio_device['index'])

    return _streams


def open_file(_file_path):
    with open(_file_path, 'rb') as handle:
        return pickle.load(handle)


def save_object(_path, _data):
    with open(_path, 'wb') as handle:
        pickle.dump(_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def record_left_audio(_attributes):
    _left_audio_data = []
    pa = pyaudio.PyAudio()
    for (mic_name, mic) in get_audio_streams(_attributes['chunk'], pa).items():
        if mic_name == 'Microphone (USB Live camera audio)':
            stream = mic

    _pa = pyaudio.PyAudio()
    while time.time() < _attributes['start_time'] + 10:
        pass
    print(f'GO! (left audio)   {time.time() - _attributes["start_time"]}')
    while time.time() < _attributes["start_time"] + _attributes['recording_time']:
        _data = stream.read(_attributes['chunk'])
        _left_audio_data.append(_data)

    print(f'Done. (left audio)   {time.time() - _attributes["start_time"]}')
    stream.stop_stream()
    stream.close()

    _path = r'D:\pythonProject\Sound_locator\temp'
    _file_name = 'left_audio_data.wav'
    _path = os.path.join(_path, _file_name)

    sf = wave.open(_path, 'wb')
    sf.setnchannels(stream._channels)
    sf.setsampwidth(pa.get_sample_size(stream._format))
    sf.setframerate(stream._rate)
    sf.writeframes(b''.join(_left_audio_data))
    sf.close()
    pa.terminate()


def record_right_audio(_attributes):
    pa = pyaudio.PyAudio()
    for (mic_name, mic) in get_audio_streams(_attributes['chunk'], pa).items():
        if mic_name != 'Microphone (USB Live camera audio)':
            stream = mic

    _right_audio_data = []
    _pa = pyaudio.PyAudio()
    while time.time() < _attributes['start_time'] + 10:
        pass
    print(f'GO! (right audio)   {time.time() - _attributes["start_time"]}')
    while time.time() < _attributes["start_time"] + _attributes['recording_time']:
        _data = stream.read(_attributes['chunk'])
        _right_audio_data.append(_data)

    print(f'Done. (right audio)   {time.time() - _attributes["start_time"]}')
    stream.stop_stream()
    stream.close()

    _path = r'D:\pythonProject\Sound_locator\temp'
    _file_name = 'right_audio_data.wav'
    _path = os.path.join(_path, _file_name)

    sf = wave.open(_path, 'wb')
    sf.setnchannels(stream._channels)
    sf.setsampwidth(pa.get_sample_size(stream._format))
    sf.setframerate(stream._rate)
    sf.writeframes(b''.join(_right_audio_data))
    sf.close()
    pa.terminate()


def get_left_video(_attributes):
    left_camera = cv.VideoCapture(0)
    _left_video_data = []

    if not left_camera.isOpened():
        print("Cannot open cameras")
        exit()
    while time.time() < _attributes["start_time"] + 10:
        pass
    print(f'GO! (left video)   {time.time() - _attributes["start_time"]}')
    while time.time() < _attributes["start_time"] + _attributes['recording_time']:
        _, left_frame = left_camera.read()

        while left_frame is None:
            _, left_frame = left_camera.read()

        _left_video_data.append(left_frame)
    print(f'Done. (left video)   {time.time() - _attributes["start_time"]}')
    _path = r'D:\pythonProject\Sound_locator\temp'
    _file_name = 'left_video_data.pkl'
    _path = os.path.join(_path, _file_name)
    save_object(_path, _left_video_data)


def get_right_video(_attributes):
    right_camera = cv.VideoCapture(1)
    _right_video_data = []

    if not right_camera.isOpened():
        print("Cannot open cameras")
        exit()
    while time.time() < _attributes["start_time"] + 10:
        pass
    print(f'GO! (right video)   {time.time() - _attributes["start_time"]}')
    while time.time() < _attributes["start_time"] + _attributes['recording_time']:
        _, right_frame = right_camera.read()

        while right_frame is None:
            _, right_frame = right_camera.read()

        _right_video_data.append(right_frame)
    print(f'Done. (right video)   {time.time() - _attributes["start_time"]}')
    _path = r'D:\pythonProject\Sound_locator\temp'
    _file_name = 'right_video_data.pkl'
    _path = os.path.join(_path, _file_name)
    save_object(_path, _right_video_data)


def plot_audio(left_channel, right_channel):
    plt.plot(left_channel, color='blue', label="Left channel")
    plt.plot(right_channel, color='red', label="Right channel", alpha=0.6)
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def clear_a_directory(folder):
    for filename in os.listdir(folder):
        _file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(_file_path) or os.path.islink(_file_path):
                os.unlink(_file_path)
            elif os.path.isdir(_file_path):
                shutil.rmtree(_file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (_file_path, e))


def fuse_audio(_left, _right):
    bitrate = _left[0]
    _left = _left[1].transpose()[0].reshape(-1, 1)
    _right = _right[1].transpose()[0].reshape(-1, 1)

    if len(_left) > len(_right):
        length = len(_right)

    else:
        length = len(_left)

    _left = _left[0:length]
    _right = _right[0:length]

    _audio = np.concatenate([_left, _right], axis=1)

    return {'bitrate': bitrate, 'audio': _audio}


if __name__ == "__main__":
    ################################################
    # Setup audio
    chunk = 1024
    ################################################

    ################################################
    go_time = time.time()
    recording_time = 20

    p1 = mp.Process(target=record_left_audio, args=({'start_time': go_time, 'recording_time': recording_time,
                                                     'chunk': chunk},))
    p2 = mp.Process(target=record_right_audio, args=({'start_time': go_time, 'recording_time': recording_time,
                                                      'chunk': chunk},))
    p3 = mp.Process(target=get_left_video, args=({'start_time': go_time, 'recording_time': recording_time},))
    p4 = mp.Process(target=get_right_video, args=({'start_time': go_time, 'recording_time': recording_time},))

    p3.start()
    p4.start()
    p1.start()
    p2.start()

    p3.join()
    p4.join()
    p1.join()
    p2.join()
    ################################################
    path = r'D:\pythonProject\Sound_locator\temp'

    left_audio = read(os.path.join(path, 'left_audio_data.wav'))
    left_video = open_file(os.path.join(path, 'left_video_data.pkl'))
    right_audio = read(os.path.join(path, 'right_audio_data.wav'))
    right_video = open_file(os.path.join(path, 'right_video_data.pkl'))

    audio = fuse_audio(left_audio, right_audio)

    left_video_data = np.array(left_video)
    right_video_data = np.array(right_video)

    plot_audio(left_audio[1], right_audio[1])

    to_save = {'audio_data': audio,
               'left_video_data': left_video_data,
               'right_video_data': right_video_data}

    file_name = f'Raw data - {time.asctime().replace(":", "_")}.pkl'
    file_path = os.path.join(os.getcwd(), 'raw data', file_name)
    save_object(file_path, to_save)
    clear_a_directory(path)

    print(f'\n\nSaved file: {file_name}')
