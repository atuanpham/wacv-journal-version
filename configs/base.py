import os


class BaseConfig(object):

    # Project root path
    ROOT_DIR = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..'
    ))

    # Configure data path
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

    # Configurations for augmentation task
    AUG_SIZE = 32
    AUG_CONFIGS = dict(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Trained-models results
    MODELS_DIR = os.path.join(ROOT_DIR, 'models')
    WEIGHTS_DIR = os.path.join(MODELS_DIR, 'weights')

