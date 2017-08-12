from skimage import io
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import numpy as np


class TiffImage(object):

    def __init__(self, path):
        self.path = path

    def convertToNumpyArray(self):
        im = io.imread(self.path)
        return im


class Augmentation(object):

    def __init__(self, train_data, label_data, aug_size=32, aug_configs=None):
        self.train_data = train_data
        self.label_data = label_data
        self.aug_size = aug_size
        self.data_gen_args = aug_configs

        if self.data_gen_args is None:
            self.data_gen_args = dict(
                rotation_range=0.2,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest'
            )

    def augment(self, batch_size=1):
        K.set_image_data_format('channels_first')
        train_gen = ImageDataGenerator(**self.data_gen_args)
        label_gen = ImageDataGenerator(**self.data_gen_args)

        for i in range(len(self.train_data)):
            train_temp = self.train_data[i].reshape(1, 1, *self.train_data[i].shape)
            label_temp = self.label_data[i].reshape(1, 1, *self.label_data[i].shape)

            train_gen.fit(train_temp, augment=True, seed=i)
            label_gen.fit(label_temp, augment=True, seed=i)

            train_flow = train_gen.flow(train_temp, batch_size=batch_size, seed=i)
            label_flow = label_gen.flow(label_temp, batch_size=batch_size, seed=i)
            self._merge_to_origin(train_flow, label_flow)

        print('Augmentation completed.')

    def _merge_to_origin(self, train_flow, label_flow):

        print('Appending additional train volume.')
        for i, aug_train in enumerate(train_flow):
            aug_train = aug_train.reshape(*aug_train.shape[1:])  # reshape from (1, 1, width, height) to (1, width, height)
            self.train_data = np.append(self.train_data, aug_train, axis=0)

            if i + 1 >= self.aug_size:
                break

        print('Appending additional train labels.')
        for i, aug_label in enumerate(label_flow):
            aug_label = aug_label.reshape(*aug_label.shape[1:])  # reshape from to (1, width, height)
            self.label_data = np.append(self.label_data, aug_label, axis=0)

            if i + 1 >= self.aug_size:
                break


class Preprocessor(object):

    def __init__(self, raw_train_data_path, raw_label_data_path, aug_size=32, aug_configs=None):
        self.raw_train_data_path = raw_train_data_path
        self.raw_label_data_path = raw_label_data_path
        self.preprocessed_train_data = None
        self.preprocessed_label_data = None
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

        # Augment train data & label data
        augmentation = Augmentation(raw_train_data, raw_label_data, self.aug_size, self.aug_configs)
        augmentation.augment()
        aug_train = augmentation.train_data
        aug_label = augmentation.label_data

        # Reshape data
        self.preprocessed_train_data = aug_train.reshape(aug_train.shape[0], 1, *aug_train.shape[1:])
        self.preprocessed_label_data = aug_label.reshape(aug_label.shape[0], 1, *aug_label.shape[1:])

    def save(self, preprocessed_train_path, preprocessed_label_path):
        np.save(preprocessed_train_path, self.preprocessed_train_data)
        np.save(preprocessed_label_path, self.preprocessed_label_data)
        print('Persisted preprocessed data.')


if __name__ == '__main__':
    raw_train_data_path = '../raw_data/train-volume.tif'
    raw_label_data_path = '../raw_data/train-labels.tif'
    preprocessed_train_path = '../preprocessed_data/train_data.npy'
    preprocessed_label_path = '../preprocessed_data/label_data.npy'

    preprocessor = Preprocessor(raw_train_data_path, raw_label_data_path, aug_size=3)

    print('Preprocessing data...')
    preprocessor.do_preprocess()
    print('Finish preprocess.')

    preprocessor.save(preprocessed_train_path, preprocessed_label_path)

