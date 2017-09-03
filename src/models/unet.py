import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D
from keras.layers.merge import concatenate, Concatenate
from keras.callbacks import ModelCheckpoint 
from .decorators import UnetDecorator
from ..exceptions import ModelError


class Unet(object):
    """
    :type input_rows: int
    :type input_columns: int
    :type model: Model
    :type weights_path: str
    :type weights_loaded: bool
    """

    def __init__(self, input_rows, input_columns, weights_path=None):
        self.input_rows = input_rows
        self.input_columns = input_columns
        self.model = None
        self.weights_path = weights_path
        self.weights_loaded = False  # used for checking whether weights are loaded

    def _get_unet_model(self):
        inputs = Input((self.input_rows, self.input_columns, 1))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    @UnetDecorator.load_model
    def train(self, train_data, mask_data, epochs=10):
        """
        :type train_data: numpy.ndarray
        :type mask_data: numpy.ndarray
        :type epochs: int
        """

        if self.weights_path is None:
            raise ModelError('Weights path is not defined.')

        model_checkpoint = ModelCheckpoint(self.weights_path, monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        self.model.fit(train_data, mask_data, batch_size=1, epochs=epochs, validation_split=0.2,
                  verbose=1, shuffle=True, callbacks=[model_checkpoint])

    @UnetDecorator.load_model
    @UnetDecorator.load_weights
    def predict(self, data, threshold=0.5, batch_size=1, verbose=1):
        """
        :type data: numpy.ndarray
        :type threshold: float
        :type batch_size: int
        :type verbose: int

        :rtype numpy.ndarray
        """

        predictions = self.model.predict(data, batch_size=batch_size, verbose=verbose)
        predictions[predictions <= threshold] = 0
        predictions[predictions > threshold] = 1
        return predictions

    @UnetDecorator.load_model
    @UnetDecorator.load_weights
    def evaluate(self, predictions, test_mask, class_index):
        """
        :type predictions: numpy.ndarray
        :type test_mask: numpy.ndarray
        :type class_index: int

        :rtype float
        """

        data_f = predictions[:, :, :, class_index].flatten()
        mask_f = test_mask[:, :, :, class_index].flatten()
        intersection = np.sum(data_f * mask_f)

        return (2 * intersection) / (np.sum(data_f) + np.sum(mask_f))

    @UnetDecorator.load_model
    @UnetDecorator.load_weights
    def evaluate_average(predictions, test_mask):
        """
        :type predictions: numpy.ndarray
        :type test_mask: numpy.ndarray

        :rtype float
        """

        n_classes = predictions.shape[-1]
        accuracies = []

        for class_index in range(n_classes):
            accuracies.append(self.evaluate(predictions, test_mask, class_index))

        return np.mean(accuracies)

