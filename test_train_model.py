import numpy as np
from src.models.unet import Unet
from configs.development import SimpleConfig

_config = SimpleConfig


if __name__ == '__main__':
    train_data = np.load(_config.PROCESSED_TRAIN_IMAGE_PATH)
    label_data = np.load(_config.PROCESSED_TRAIN_LABEL_PATH)

    unet = Unet(512, 512)
    unet.train(train_data, label_data)
