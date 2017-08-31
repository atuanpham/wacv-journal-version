from functools import wraps
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D
from keras.layers.merge import concatenate, Concatenate
from keras.callbacks import ModelCheckpoint 
from ..exceptions import ModelError


class Unet(object):

    def __init__(self, input_rows, input_columns, weights_path=None):
        self.input_rows = input_rows
        self.input_columns = input_columns
        self.model = None
        self.weights_path = weights_path
        self.weights_loaded = False

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

    def model_required(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.model is None:
                self.model = self._get_unet_model()
            return func(*args, **kwargs)
        return wrapper

    @model_required
    def weights_required(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.weights_path is None:
                raise ModelError('Weights path is not defined.')
            if self.weights_loaded == False:
                self.model.load_weights(self.weights_path)
                self.weights_loaded = True
            return func(*args, **kwargs)
        return wrapper

    @model_required
    def train(self, train_data, mask_data, epochs=10):
        if self.weights_path is None:
            raise ModelError('Weights path is not defined.')

        model_checkpoint = ModelCheckpoint(self.weights_path, monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        self.model.fit(train_data, mask_data, batch_size=1, epochs=epochs, validation_split=0.2,
                  verbose=1, shuffle=True, callbacks=[model_checkpoint])

    @weights_required
    def predict(self, data, batch_size=1, verbose=1):
        predictions = self.model.predict(data, batch_size=batch_size, verbose=verbose)
        return predictions

    @weights_required
    def evaluate(self, test_data, test_mask, batch_size=1, verbose=1):
        score, acc = model.evaluate(test_data, test_mask, batch_size=batch_size, verbose=verbose)
        return score, acc

