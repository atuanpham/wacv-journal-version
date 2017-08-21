import os


def transform_raw_data(data):
    return data.reshape(data.shape[0], *data.shape[1:], 1)


def get_direct_directories(path):
    dir_walker = os.walk(path)
    return next(dir_walker)[1]

