import numpy as np
from ..utils.tiff_image import TiffImage
from ..utils import misc
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
        self.aug_configs = aug_configs if aug_configs else dict()

    def do_preprocess(self):
        # Read Tiff images and convert to numpy arrays
        raw_train_data = TiffImage(self.raw_train_data_path).convert_to_numpy_array()
        raw_label_data = TiffImage(self.raw_label_data_path).convert_to_numpy_array()
        raw_test_data = TiffImage(self.raw_test_data_path).convert_to_numpy_array()

        # Transform raw data from (samples, width, height) to (samples, width, height, channels)
        raw_train_data = misc.transform_raw_data(raw_train_data)
        raw_label_data = misc.transform_raw_data(raw_label_data)

        # Augment train data & label data
        augmentation = Augmentation(raw_train_data, raw_label_data, self.aug_size, self.aug_configs)
        augmentation.augment()

        self.preprocessed_train_data = augmentation.train_data
        self.preprocessed_label_data = augmentation.label_data

        # Reshape test data also
        self.preprocessed_test_data = misc.transform_raw_data(raw_test_data)

    def save(self, preprocessed_train_path, preprocessed_label_path, preprocessed_test_path):
        np.save(preprocessed_train_path, self.preprocessed_train_data)
        np.save(preprocessed_label_path, self.preprocessed_label_data)
        np.save(preprocessed_test_path, self.preprocessed_test_data)
        print('Persisted preprocessed data.')
