import numpy as np
import keras.backend as K
from src.model import Unet


if __name__ == '__main__':

    # K.set_image_data_format('channels_first')

    train_data = np.load('./preprocessed_data/train_data.npy')
    label_data = np.load('./preprocessed_data/label_data.npy')

    train_data = train_data.reshape(120, 512, 512, 1)
    label_data = label_data.reshape(120, 512, 512, 1)

    unet = Unet(512, 512)
    unet.train(train_data, label_data)
