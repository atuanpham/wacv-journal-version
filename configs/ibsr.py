import os
from src.config.base import BaseConfig


class IBSRConfig(BaseConfig):

    # Raw data path
    RAW_TRAIN_DIR = os.path.join(BaseConfig.RAW_DATA_DIR, 'IBSR_18/train')
    RAW_TEST_DIR = os.path.join(BaseConfig.RAW_DATA_DIR, 'IBSR_18/test')

    # Path to processed images
    PROCESSED_TRAIN_DATA_DIR = os.path.join(BaseConfig.PROCESSED_DATA_DIR, 'train')
    PROCESSED_TEST_DATA_DIR = os.path.join(BaseConfig.PROCESSED_DATA_DIR, 'test')

    # Rules of Nifti files
    POSTFIX_DATA_FILE = '_ana_strip.nii.gz'
    POSTFIX_MASK_DATA_FILE = '_segTRI_fill_ana.nii.gz'

    # Results path
    WEIGHTS_PATH = os.path.join(BaseConfig.RESULTS_PATH, 'weights')

    # Augmentation config
    # AUG_SIZE = 30

    # Train configs
    EPOCHS = 10 

