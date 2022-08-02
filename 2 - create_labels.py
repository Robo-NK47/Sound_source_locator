from os import listdir
from os.path import isfile, join
import os
import pickle
from tqdm.auto import tqdm


def open_file(_file_path):
    with open(_file_path, 'rb') as handle:
        return pickle.load(handle)


def detection_to_xy_label(_data):
    labels = []
    for _detection in _data:
        if len(_detection.index) > 0:
            _x = ((_detection['xmin'] + _detection['xmax']) * 0.5).values[0]
            _y = ((_detection['ymin'] + _detection['ymax']) * 0.5).values[0]
            labels.append([_x, _y])
    return labels


detection_path = os.path.join(os.getcwd(), 'detection data')
labeled_path = os.path.join(os.getcwd(), 'labeled data')
list_of_raw_data_files = [f for f in listdir(detection_path) if isfile(join(detection_path, f))]
distance_between_cameras = 140  # in [mm]
distance_lens_to_sensor = 20  # in [mm]

for raw_data_file_name in tqdm(list_of_raw_data_files):
    file_path = os.path.join(detection_path, raw_data_file_name)
    data = open_file(file_path)

    left_camera_detections = detection_to_xy_label(data['left_video_data'])
    right_camera_detections = detection_to_xy_label(data['right_video_data'])

    detection = []

    for left_data, right_data in zip(left_camera_detections, right_camera_detections):
        x_left = left_data[0]
        x_right = right_data[0]

        y_left = left_data[1]
        y_right = right_data[1]

        z = (distance_between_cameras * distance_lens_to_sensor) / (x_right - x_left)

        x = (x_left + x_right) * 0.5
        y = (y_left + y_right) * 0.5

        detection.append([x, y, z])

    del data['left_video_data'], data['right_video_data']
    data['labels'] = detection

    file_path = os.path.join(labeled_path, raw_data_file_name)
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
