from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose
from keras.layers.merge import Concatenate, concatenate
from keras.callbacks import ModelCheckpoint 

class Unet(object):

    def __init__(self, input_rows, input_columns):
        self.input_rows = input_rows
        self.input_columns = input_columns

    def _get_model(self):

        # inputs = Input((1, self.input_rows, self.input_columns))

        # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # print('conv1 shape:', conv1.shape)
        # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        # print('conv1 shape:', conv1.shape)
        # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # print('pool1 shape:', pool1.shape)

        # conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        # print('conv2 shape:', conv2.shape)
        # conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        # print('conv2 shape:', conv2.shape)
        # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # print('pool2 hape:', pool2.shape)

        # conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        # print('conv3 shape:', conv3.shape)
        # conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        # print('conv3 shape:', conv3.shape)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # print('pool3 shape:', pool3.shape)

        # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        # print('conv4 shape:', conv4.shape)
        # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        # print('conv4 shape:', conv4.shape)
        # drop4 = Dropout(0.5)(conv4)
        # print('drop4 shape:', drop4.shape)
        # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        # print('pool4 shape:', pool4.shape)

        # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        # print('conv5 shape:', conv5.shape)
        # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        # print('conv5 shape:', conv5.shape)
        # drop5 = Dropout(0.5)(conv5)
        # print('drop5 shape:', drop5.shape)

        # up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(drop5)
        # )
        # print('up6 shape:', up6.shape)
        # merge6 = Concatenate(axis=3)([up6, drop4])
        # print('merge6 shape:', merge6.shape)
        # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        # print('conv6 shape:', conv6.shape)
        # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        # print('conv6 shape:', conv6.shape)

        # up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(conv6)
        # )
        # print('up7 shape:', up7.shape)
        # merge7 = Concatenate(axis=3)([up7, conv3])
        # print('merge7 shape:', merge7.shape)
        # conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        # print('conv7 shape:', conv7.shape)
        # conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        # print('conv7 shape:', conv7.shape)

        # up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(conv7)
        # )
        # print('up8 shape:', up8.shape)
        # merge8 = Concatenate(axis=3)([up8, conv2])
        # print('merge8 shape:', merge8.shape)
        # conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        # print('conv8 shape:', conv8.shape)
        # conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        # print('conv8 shape:', conv8.shape)

        # up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(conv8)
        # )
        # print('up9 shape:', up9.shape)
        # merge9 = Concatenate(axis=3)([up9, conv1])
        # print('merge9 shape:', merge9.shape)
        # conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        # print('conv9 shape:', conv9.shape)
        # conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # print('conv9 shape:', conv9.shape)
        # conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # print('conv9 shape:', conv9.shape)
        # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        # print('conv10 shape:', conv10.shape)

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

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self, train_data, label_data):
        model = self._get_model()
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)

        print('Fitting model...')
        model.fit(train_data, label_data, batch_size=1, epochs=10,
                  verbose=1, shuffle=True, callbacks=[model_checkpoint])

