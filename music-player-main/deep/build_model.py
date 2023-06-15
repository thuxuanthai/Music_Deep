# # Đây là file định nghĩa model(mạng) và các layer
# from keras.models import Sequential
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
# from keras.layers.core import Activation
# from keras.layers.core import Flatten
# from keras.layers.core import Dense
# from keras import backend as K

# class Build_model:
#     @staticmethod
#     def build(width, height, depth, classes):
#         # Khởi tạo model
#         model = Sequential()
#         input_shape = (height, width, depth)

#         # sử dụng 'channels-first' để input shape
#         if K.image_data_format() == 'channels_first':
#             input_shape = (depth, height, width)

#         # ########
#         model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
#         model.add(Activation('relu'))
#         model.add(Conv2D(32, (3, 3), padding='same'))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))

#         model.add(Conv2D(64, (3, 3), padding='same'))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))

#         model.add(Flatten())
#         model.add(Dense(512))
#         model.add(Activation('relu'))

#         # Softmax classifier
#         model.add(Dense(classes))
#         model.add(Activation('softmax'))

#         # Trả về model (Mạng CNN lenet)
#         return model



# Đây là file định nghĩa model(mạng) và các layer
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2


class Build_model:
    @staticmethod
    def build(width, height, depth, classes):
        # Khởi tạo model
        model = Sequential()
        input_shape = (height, width, depth)

        # kiểm tra định dạng dữ liệu ảnh
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        # ########
        # %%
            # MobileNetV2 => để phân loại ảnh, để giảm số lượng tham số của mạng và gia tăng tốc độ tính toán.
            base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)

            for layer in base_model.layers:
                layer.trainable = False         # => để giữ nguyên các trọng số của mô hình đã được huấn luyện trước đó

            model.add(base_model)

        # Lớp CONV: phép tích chập (convolution) trên ảnh đầu vào với các filter, 
        #           -> để trích xuất các đặc trưng quan trọng của ảnh.
        # Lớp RELU: loại bỏ các giá trị âm và giữ lại các giá trị dương, 
        #           -> giúp tăng tốc độ tính toán và giảm thiểu vấn đề biến mất đạo hàm.
        # Lớp POOL: -> giảm kích thước của feature maps và giảm độ phức tạp tính toán của mạng.
        
        # Việc lặp lại này giúp trích xuất các đặc trưng phức tạp của ảnh và giảm kích thước của 
        #       feature maps đến mức có thể chứa thông tin quan trọng nhất của ảnh.
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))       

        # Thứ hai: CONV => RELU => POOL layers
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten  => Làm phẳng rồi đưa vào lớp FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))

        # Dense: thực hiện phân loại đối tượng hoặc dự đoán kết quả của mô hình
        # Softmax classifier   =>  đưa ra dự đoán xác suất cho từng lớp đầu ra
        model.add(Dense(classes))
        model.add(Activation('softmax'))  

        # Trả về model (Mạng CNN build_model)
        return model