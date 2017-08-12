import numpy as np
from .utils import TiffImage
from .augmentation import Augmentation


class Preprocessor(object):

    def __init__(self, raw_train_data_path, raw_label_data_path, raw_test_data_path, aug_size=32, aug_configs=None):
        self.raw_train_data_path = raw_train_data_path
        self.raw_label_data_path = raw_label_data_path
        self.raw_test_data_path = raw_test_data_path
        self.preprocessed_train_data = None
        self.preprocessed_label_data = None
        self.preprocessed_test_data = None
        self.aug_size = aug_size
        self.aug_configs = aug_configs

        if self.aug_configs is None:
            self.aug_configs = dict(
                rotation_range=0.2,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest'
            )

    def do_preprocess(self):
        # Read Tiff images and convert to numpy arrays
        raw_train_data = TiffImage(self.raw_train_data_path).convertToNumpyArray()
        raw_label_data = TiffImage(self.raw_label_data_path).convertToNumpyArray()
        raw_test_data = TiffImage(self.raw_test_data_path).convertToNumpyArray()

        # Augment train data & label data
        augmentation = Augmentation(raw_train_data, raw_label_data, self.aug_size, self.aug_configs)
        augmentation.augment()
        aug_train = augmentation.train_data
        aug_label = augmentation.label_data

        # Reshape data
        self.preprocessed_train_data = aug_train.reshape(aug_train.shape[0], 1, *aug_train.shape[1:])
        self.preprocessed_label_data = aug_label.reshape(aug_label.shape[0], 1, *aug_label.shape[1:])

        # Reshape test data also
        self.preprocessed_test_data = raw_test_data.reshape(raw_test_data.shape[0], 1, *raw_test_data.shape[1:])

    def save(self, preprocessed_train_path, preprocessed_label_path, preprocessed_test_path):
        np.save(preprocessed_train_path, self.preprocessed_train_data)
        np.save(preprocessed_label_path, self.preprocessed_label_data)
        np.save(preprocessed_test_path, self.preprocessed_test_data)
        print('Persisted preprocessed data.')

