import os
import numpy as np
from ..utils import misc


class DataUtils(object):
    """
    :type processed_train_data_dir: str
    :type processed_test_data_dir: str
    :type train_patients: list[str]
    :type test_patients: list[str]
    :type train_data: np.ndarray
    :type train_mask: np.ndarray
    :type test_data: np.ndarray
    :type test_mask: np.ndarray
    """

    __data_postfix = '_data.npy'
    __mask_postfix = '_mask.npy'

    def __init__(self, processed_train_data_dir, processed_test_data_dir):
        self.processed_train_data_dir = processed_train_data_dir
        self.processed_test_data_dir = processed_test_data_dir

        self.train_patients = misc.get_direct_directories(self.processed_train_data_dir)
        self.test_patients = misc.get_direct_directories(self.processed_test_data_dir)

        self.train_data = None  # type: np.ndarray
        self.train_mask = None  # type: np.ndarray
        self.test_data = None  # type: np.ndarray
        self.test_mask = None  # type: np.ndarray

    def get_train_data(self):
        # don't need to read data again
        if self.train_data is not None and self.train_mask is not None:
            return (self.train_data, self.train_mask)

        for patient in self.train_patients:
            self.__append_train_data(patient)

        return (self.train_data, self.train_mask)

    def get_test_data(self):
        # don't need to read data again
        if self.test_data is not None and self.test_mask is not None:
            return (self.test_data, self.test_mask)

        for patient in self.test_patients:
            self.__append_test_data(patient)

        return (self.test_data, self.test_mask)

    def __append_train_data(self, patient):
        data_path = os.path.join(self.processed_train_data_dir, patient, '{}{}'.format(patient, self.__data_postfix))
        mask_path = os.path.join(self.processed_train_data_dir, patient, '{}{}'.format(patient, self.__mask_postfix))

        data = np.load(data_path)
        mask = np.load(mask_path)

        if self.train_data is None:
            self.train_data = data
        else:
            self.train_data = np.append(self.train_data, data, axis=0)

        if self.train_mask is None:
            self.train_mask = mask
        else:
            self.train_mask = np.append(self.train_mask, mask, axis=0)

    def __append_test_data(self, patient):
        data_path = os.path.join(self.processed_test_data_dir, patient, '{}{}'.format(patient, self.__data_postfix))
        mask_path = os.path.join(self.processed_test_data_dir, patient, '{}{}'.format(patient, self.__mask_postfix))

        data = np.load(data_path)
        mask = np.load(mask_path)

        if self.test_data is None:
            self.test_data = data
        else:
            self.test_data = np.append(self.test_data, data, axis=0)

        if self.test_mask is None:
            self.test_mask = mask
        else:
            self.test_mask = np.append(self.test_mask, mask, axis=0)


class DataLabel(object):

    @staticmethod
    def extract_mask_labels(mask, number_of_classes):
        output_shape = (mask.shape[0], mask.shape[1], mask.shape[2], number_of_classes)
        extracted_mask = np.zeros(output_shape)

        for index, slice_ in enumerate(mask):
            for class_index in range(number_of_classes):
                class_label = class_index + 1
                extracted_mask[index, :, :, class_index][mask[index, :, :, 0] == class_label] = 1

        return extracted_mask

    @staticmethod
    def merge_mask_labels(extracted_mask):
        output_shape = (extracted_mask.shape[0], extracted_mask.shape[1], extracted_mask.shape[2], 1)
        mask = np.zeros(output_shape)

        for index, slice_ in enumerate(extracted_mask):
            for class_index in range(extracted_mask.shape[3]):
                label = class_index + 1
                mask[index, :, :, 0][extracted_mask[index, :, :, class_index] == 1] = label

        return mask

