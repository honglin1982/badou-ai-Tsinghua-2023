# import cv2.dnn
import numpy as np
from typing import List
from keras import layers
from keras.models import Model
from keras.layers import Dense, Activation, MaxPooling2D, AveragePooling2D, BatchNormalization, Conv2D, Input, \
    GlobalAveragePooling2D
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
# from keras.preprocessing import image
from keras.utils import image_utils


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=(1, 1),
              padding="same",
              name=None):
    if name is not None:
        bn_name = f'{name}_bn'
        conv_name = f'{name}_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def inception_v3(input_shape: List[int] = [299, 299, 3], classes=1000):
    img_input = Input(shape=input_shape)

    x = conv2d_bn(img_input, filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid')  # 149, 149, 32
    x = conv2d_bn(x, filters=32, kernel_size=(3, 3), padding='valid')  # 147, 147, 32
    x = conv2d_bn(x, filters=64, kernel_size=(3, 3))  # 147, 147, 64
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)  # 73, 73, 64

    x = conv2d_bn(x, filters=80, kernel_size=(1, 1), padding='valid')  # 73, 73, 80
    x = conv2d_bn(x, filters=192, kernel_size=(3, 3), padding='valid')  # 71， 71， 192
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)  # 35， 35， 192

    # --------------------------------#
    #   Block1 35x35
    # --------------------------------#
    # Block1 part1
    # 35 x 35 x 192 -> 35 x 35 x 256
    branch_1x1 = conv2d_bn(x, 64, kernel_size=(1, 1))

    branch_5x5 = conv2d_bn(x, 48, kernel_size=(1, 1))
    branch_5x5 = conv2d_bn(branch_5x5, 64, kernel_size=(5, 5))

    branch3x3dbl = conv2d_bn(x, 64, kernel_size=(1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, kernel_size=(3, 3))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, kernel_size=(3, 3))

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, kernel_size=(1, 1))

    # 64+64+96+32 = 256  nhwc-0123
    x = layers.concatenate(
        [branch_1x1, branch_5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed0'
    )

    # Block1 part2
    # 35 x 35 x 256 -> 35 x 35 x 288
    branch2_1x1 = conv2d_bn(x, 64, kernel_size=(1, 1))

    branch2_5x5 = conv2d_bn(x, 48, kernel_size=(1, 1))
    branch2_5x5 = conv2d_bn(branch2_5x5, 64, kernel_size=(5, 5))

    branch2_3x3 = conv2d_bn(x, 64, kernel_size=(1, 1))
    branch2_3x3 = conv2d_bn(branch2_3x3, 96, kernel_size=(3, 3))
    branch2_3x3 = conv2d_bn(branch2_3x3, 96, kernel_size=(3, 3))

    branch2_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch2_pool = conv2d_bn(branch2_pool, 64, kernel_size=(1, 1))
    # 64+64+96+64 = 288
    x = layers.concatenate(
        [branch2_1x1, branch2_5x5, branch2_3x3, branch2_pool],
        axis=3,
        name='mixed1'
    )

    # Block1 part3
    # 35 x 35 x 288 -> 35 x 35 x 288
    branch2_1x1 = conv2d_bn(x, 64, kernel_size=(1, 1))
    branch2_5x5 = conv2d_bn(x, 48, kernel_size=(1, 1))
    branch2_5x5 = conv2d_bn(branch2_5x5, 64, kernel_size=(5, 5))
    branch2_3x3 = conv2d_bn(x, 64, kernel_size=(1, 1))
    branch2_3x3 = conv2d_bn(branch2_3x3, 96, kernel_size=(3, 3))
    branch2_3x3 = conv2d_bn(branch2_3x3, 96, kernel_size=(3, 3))
    branch2_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch2_pool = conv2d_bn(branch2_pool, 64, kernel_size=(1, 1))
    # 64+64+96+64 = 288
    x = layers.concatenate(
        [branch2_1x1, branch2_5x5, branch2_3x3, branch2_pool],
        axis=3,
        name='mixed2'
    )
    #--------------------------------#
    #   Block2 17x17
    #--------------------------------#
    # Block2 part1
    # 35 x 35 x 288 -> 17 x 17 x 768
    branch3_3x3c = conv2d_bn(x, 384, kernel_size=(3, 3), strides=(2, 2), padding='valid')  # 17 x 17 x 384

    branch3_3x3 = conv2d_bn(x, 64, kernel_size=(1, 1))  # 35,35,64
    branch3_3x3 = conv2d_bn(branch3_3x3, 96, kernel_size=(3, 3))  # 35,35,96
    branch3_3x3 = conv2d_bn(branch3_3x3, 96, kernel_size=(3, 3), strides=(2, 2), padding='valid')  # 17 x 17 x 96

    branch3_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)  # 17, 17, 288
    x = layers.concatenate(
        [branch3_3x3c, branch3_3x3, branch3_pool],
        axis=3,
        name='mixed3'
    )
    # Block2 part2
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, kernel_size=(1, 1))
    branch7x7 = conv2d_bn(x, 128, kernel_size=(1, 1))
    branch7x7 = conv2d_bn(branch7x7, 128, kernel_size=(1, 7))
    branch7x7 = conv2d_bn(branch7x7, 192, kernel_size=(7, 1))
    branch7x7dbl = conv2d_bn(x, 128, kernel_size=(1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, kernel_size=(7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, kernel_size=(1, 7))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, kernel_size=(7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(1, 7))
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, kernel_size=(1, 1))

    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed4'
    )
    # Block2 part3 and part4
    # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, kernel_size=(1, 1))

        branch7x7 = conv2d_bn(x, 160, kernel_size=(1, 1))
        branch7x7 = conv2d_bn(branch7x7, 160, kernel_size=(1, 7))
        branch7x7 = conv2d_bn(branch7x7, 192, kernel_size=(7, 1))

        branch7x7dbl = conv2d_bn(x, 160, kernel_size=(1, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, kernel_size=(7, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, kernel_size=(1, 7))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, kernel_size=(7, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(1, 7))

        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, kernel_size=(1, 1))
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed' + str(5 + i)
        )

    # Block2 part5
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, kernel_size=(1, 1))

    branch7x7 = conv2d_bn(x, 192, kernel_size=(1, 1))
    branch7x7 = conv2d_bn(branch7x7, 192, kernel_size=(1, 7))
    branch7x7 = conv2d_bn(branch7x7, 192, kernel_size=(7, 1))

    branch7x7dbl = conv2d_bn(x, 192, kernel_size=(1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(1, 7))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(1, 7))

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, kernel_size=(1, 1))
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed7'
    )

    # --------------------------------#
    #   Block3 8x8
    # --------------------------------#
    # Block3 part1
    # 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, kernel_size=(1, 1))
    branch3x3 = conv2d_bn(branch3x3, 320, kernel_size=(3, 3), strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, kernel_size=(1, 1))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, kernel_size=(1, 7))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, kernel_size=(7, 1))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, kernel_size=(3, 3), strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=3,
        name='mixed8'
    )

    # Block3 part2 part3
    # 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, kernel_size=(1, 1))

        branch3x3 = conv2d_bn(x, 384, kernel_size=(1, 1))
        branch3x3 = conv2d_bn(branch3x3, 384, kernel_size=(1, 3))
        branch3x3 = conv2d_bn(branch3x3, 384, kernel_size=(3, 1))
        branch3x3 = layers.concatenate(
            [branch1x1, branch3x3],
            axis=3,
            name='mixed9_' + str(i)
        )
        branch3x3dbl = conv2d_bn(x, 448, kernel_size=(1, 1))
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, kernel_size=(3, 3))
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, kernel_size=(1, 3))
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, kernel_size=(3, 1))
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2],
            axis=3
        )
        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, kernel_size=(1, 1))
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed' + str(9 + i)
        )
    # 平均池化后全连接。
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    model = Model(inputs, x, name='inception_v3')
    return model


def diy_preprocess_input(x):
    """ 将数值 变为均值为0，标准差为1的正态分布， """
    x /= 255
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':

    from inceptionV3 import InceptionV3

    # model = inception_v3()
    model = InceptionV3()
    model.load_weights(r'inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
    img = image_utils.load_img(path='../../file/elephant.jpg', target_size=(299, 299))
    img = image_utils.img_to_array(img)
    print(img.shape)
    img = np.expand_dims(img, axis=0)
    x = diy_preprocess_input(img)
    preds = model.predict(x)

    print('Predicted:', decode_predictions(preds))
    pass

