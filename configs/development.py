import os
from .base import BaseConfig


class SimpleConfig(BaseConfig):

    # Path to raw images
    RAW_TRAIN_IMAGE_PATH = os.path.join(BaseConfig.RAW_DATA_DIR, 'train-volume.tif')
    RAW_TRAIN_LABEL_IMAGE_PATH = os.path.join(BaseConfig.RAW_DATA_DIR, 'train-labels.tif') 
    RAW_TEST_IMAGE_PATH = os.path.join(BaseConfig.RAW_DATA_DIR, 'test-volume.tif')

    # Path to processed images
    PROCESSED_TRAIN_IMAGE_PATH = os.path.join(BaseConfig.PROCESSED_DATA_DIR, 'train.npy')
    PROCESSED_TRAIN_LABEL_PATH = os.path.join(BaseConfig.PROCESSED_DATA_DIR, 'train-labels.npy')
    PROCESSED_TEST_PATH = os.path.join(BaseConfig.PROCESSED_DATA_DIR, 'test.npy')

    # Augmentation config
    AUG_SIZE = 30

