"""
-------------------------------------------------------------
   ResNet50的网络部分
-------------------------------------------------------------
"""
import numpy as np
import keras.backend as K
from keras.layers import (Dense, Input, ZeroPadding2D, AveragePooling2D, Conv2D, MaxPooling2D, Activation,
                          BatchNormalization, Flatten)
from keras.models import Model
from keras import layers

K.set_image_data_format('channels_last')


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = f'res{stage}{block}_branch'
    bn_name_base = f'bn{stage}{block}_branch'
    # [56, 56, 64]
    x = Conv2D(filters1, (1, 1), strides=strides, name=f'{conv_name_base}2a')(input_tensor)
    x = BatchNormalization(name=f'{bn_name_base}2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size=kernel_size, padding='same', name=f'{conv_name_base}2b')(x)
    x = BatchNormalization(name=f'{bn_name_base}2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=f'{conv_name_base}2c')(x)
    x = BatchNormalization(name=f'{bn_name_base}2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=f'{conv_name_base}1')(input_tensor)
    shortcut = BatchNormalization(name=f'{bn_name_base}1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = f'res{stage}{block}_branch'
    bn_name_base = f'bn{stage}{block}_branch'

    x = Conv2D(filters1, (1, 1), name=f'{conv_name_base}2a')(input_tensor)
    x = BatchNormalization(name=f'{bn_name_base}2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=f'{conv_name_base}2b')(x)
    x = BatchNormalization(name=f'{bn_name_base}2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=f'{conv_name_base}2c')(x)
    x = BatchNormalization(name=f'{bn_name_base}2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def resnet50(input_tensor=None, input_shape=None, classes=1000, **kwargs):

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    elif not K.is_keras_tensor(input_tensor):
        img_input = Input(tensor=input_tensor, shape=input_shape)
    else:
        img_input = input_tensor

    # img_input (224, 224, 3)
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)  # 112, 112, 64
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    # 55, 55, 64
    x = conv_block(x, 3, filters=[64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, filters=[64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, filters=[64, 64, 256], stage=2, block='c')
    # 结果：[55, 55, 256]
    x = conv_block(x, 3, filters=[128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, filters=[128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, filters=[128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, filters=[128, 128, 512], stage=3, block='d')
    # 结果：[28, 28, 512]
    x = conv_block(x, 3, filters=[256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, filters=[256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, filters=[256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, filters=[256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, filters=[256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, filters=[256, 256, 1024], stage=4, block='f')
    # 结果：[14, 14, 1024]
    x = conv_block(x, 3, filters=[512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, filters=[512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, filters=[512, 512, 2048], stage=5, block='c')
    # 结果：[7, 7, 2048]
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    # 结果：[1, 1, 2048]
    x = Flatten()(x)
    # 结果：(None, 2048)
    x = Dense(classes, activation='softmax', name='fc1000')(x)
    # 1000
    model = Model(img_input, x, name='resnet50')
    # model.load_weights(r'../../代码/resnet50_tf/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    model.load_weights('/Users/dingkou/PycharmProjects/pythonProject/NlpPratice/pratice01/week11/resnet50_keras/nets/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    return model






