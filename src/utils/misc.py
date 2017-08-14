def transform_raw_data(data):
    return data.reshape(data.shape[0], *data.shape[1:], 1)