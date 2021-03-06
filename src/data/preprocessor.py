import os
import errno
import numpy as np
from .utils import DataLabel
from ..utils import misc
from ..utils.neuro import NiftiImage


class Preprocessor(object):
    """
    :type raw_train_data_dir: str
    :type raw_test_data_dir: str
    :type processed_train_data_dir: str
    :type processed_test_data_dir: str
    :type postfix_data_file: str
    :type postfix_mask_data_file: str
    :type train_patients: list[str]
    :type test_patients: list[str]
    :type transpose: list[int]
    """

    def __init__(self, raw_train_data_dir, raw_test_data_dir,
                 processed_train_data_dir, processed_test_data_dir,
                 postfix_data_file, postfix_mask_data_file, transpose=None):

        self.raw_train_data_dir = raw_train_data_dir
        self.raw_test_data_dir = raw_test_data_dir

        self.processed_train_data_dir = processed_train_data_dir
        self.processed_test_data_dir = processed_test_data_dir

        self.postfix_data_file = postfix_data_file
        self.postfix_mask_data_file = postfix_mask_data_file

        self.train_patients = []
        self.test_patients = []

        self.transpose = transpose

    def do_preprocess(self):
        self.train_patients = misc.get_direct_directories(self.raw_train_data_dir)
        self.test_patients = misc.get_direct_directories(self.raw_test_data_dir)

        for patient in self.train_patients:  # type: str
            self._process_patient(patient, self.raw_train_data_dir, self.processed_train_data_dir)

        for patient in self.test_patients:  #type: str
            self._process_patient(patient, self.raw_test_data_dir, self.processed_test_data_dir)

    def _process_patient(self, patient, data_dir, processed_data_dir):
        """Read Nifti image and save as a numpy array
        
        :type patient: str
        :type data_dir: str
        :type processed_data_dir: str
        """

        data_file = '{}{}'.format(patient, self.postfix_data_file)
        mask_data_file = '{}{}'.format(patient, self.postfix_mask_data_file)
        data_file_path = os.path.join(data_dir, patient, data_file)
        mask_data_file_path = os.path.join(data_dir, patient, mask_data_file)

        data = NiftiImage(data_file_path).convert_to_numpy_array()
        mask_data = NiftiImage(mask_data_file_path).convert_to_numpy_array()

        if self.transpose is not None:
            data = np.transpose(data, self.transpose)
            mask_data = np.transpose(mask_data, self.transpose)

        mask_data = DataLabel.extract_mask_labels(mask_data, len(np.unique(mask_data)) - 1)

        processed_data_file_name = '{}_data.npy'.format(patient)
        processed_mask_data_file_name = '{}_mask.npy'.format(patient)

        # Create directory containing processed data if is doesn't exist
        try:
            os.makedirs(os.path.join(processed_data_dir, patient))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        np.save(os.path.join(processed_data_dir, patient, processed_data_file_name), data)
        np.save(os.path.join(processed_data_dir, patient, processed_mask_data_file_name), mask_data)

