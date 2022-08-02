import cv2
import pydub
import numpy as np
import pickle


def open_file(_file_path):
    with open(_file_path, 'rb') as _handle:
        return pickle.load(_handle)


def numpy_to_mp4(_data):
    width = _data.shape[2]
    height = _data.shape[1]
    fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter('test.mp4', fourcc, float(fps), (width, height))

    for frame in _data:
        video.write(frame)

    video.release()


def write(f, sr, x, normalized):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")


audio = np.array(open_file(r'D:\pythonProject\Sound_locator\temp\left_audio_data.pkl')).reshape(-1, 2)
video = np.array(open_file(r'D:\pythonProject\Sound_locator\temp\left_video_data.pkl'))

write('out2.mp3', 44100, audio, True)
numpy_to_mp4(video)


