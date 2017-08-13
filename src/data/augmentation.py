import keras.backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


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

