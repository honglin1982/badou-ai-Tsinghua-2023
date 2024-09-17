# -------------------------------------------------------------#
#   ResNet50的网络部分
# -------------------------------------------------------------#
from __future__ import print_function

import numpy as np
from keras import layers

from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model

from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[224, 224, 3], classes=1000):
    img_input = Input(shape=input_shape)
    img_data = ZeroPadding2D((3, 3))(img_input)

    img_data = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(img_data)
    img_data = BatchNormalization(name='bn_conv1')(img_data)
    img_data = Activation('relu')(img_data)
    img_data = MaxPooling2D((3, 3), strides=(2, 2))(img_data)

    img_data = conv_block(img_data, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    img_data = identity_block(img_data, 3, [64, 64, 256], stage=2, block='b')
    img_data = identity_block(img_data, 3, [64, 64, 256], stage=2, block='c')

    img_data = conv_block(img_data, 3, [128, 128, 512], stage=3, block='a')
    img_data = identity_block(img_data, 3, [128, 128, 512], stage=3, block='b')
    img_data = identity_block(img_data, 3, [128, 128, 512], stage=3, block='c')
    img_data = identity_block(img_data, 3, [128, 128, 512], stage=3, block='d')

    img_data = conv_block(img_data, 3, [256, 256, 1024], stage=4, block='a')
    img_data = identity_block(img_data, 3, [256, 256, 1024], stage=4, block='b')
    img_data = identity_block(img_data, 3, [256, 256, 1024], stage=4, block='c')
    img_data = identity_block(img_data, 3, [256, 256, 1024], stage=4, block='d')
    img_data = identity_block(img_data, 3, [256, 256, 1024], stage=4, block='e')
    img_data = identity_block(img_data, 3, [256, 256, 1024], stage=4, block='f')

    img_data = conv_block(img_data, 3, [512, 512, 2048], stage=5, block='a')
    img_data = identity_block(img_data, 3, [512, 512, 2048], stage=5, block='b')
    img_data = identity_block(img_data, 3, [512, 512, 2048], stage=5, block='c')

    img_data = AveragePooling2D((7, 7), name='avg_pool')(img_data)

    img_data = Flatten()(img_data)
    img_data = Dense(classes, activation='softmax', name='fc1000')(img_data)

    resnet_model = Model(img_input, img_data, name='resnet50')
    resnet_model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    return resnet_model


if __name__ == '__main__':
    model = ResNet50()
    model.summary()
    # img_path = 'elephant.jpg'
    img_path = '1.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
