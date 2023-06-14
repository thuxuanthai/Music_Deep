# Đây là file định nghĩa model(mạng) và các layer
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class Build_model:
    @staticmethod
    def build(width, height, depth, classes):
        # Khởi tạo model
        model = Sequential()
        input_shape = (height, width, depth)

        # sử dụng 'channels-first' để input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        # ########
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))

        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        # Trả về model (Mạng CNN lenet)
        return model
