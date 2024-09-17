#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date 2023/7/24
@author: 81-yuhaiyang

"""
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image

from config import Size
from utils.keras_utils import KerasUtils


# noinspection DuplicatedCode
def model(shape=None, classes=1000):
    if shape is None:
        shape = [299, 299, 3]

    img_tensor = Input(shape=shape)
    tensor_x = KerasUtils.conv2d_bn(img_tensor, 32, size=(3, 3), strides=(2, 2), padding='valid')  # 149* 149 * 32
    tensor_x = KerasUtils.conv2d_bn(tensor_x, 32, size=(3, 3), padding='valid')  # 147 * 147 * 32
    tensor_x = KerasUtils.conv2d_bn(tensor_x, 64, size=(3, 3))  # 147 * 147 * 64
    tensor_x = layers.MaxPooling2D((3, 3), strides=(2, 2))(tensor_x)  # 73 * 73 * 64

    tensor_x = KerasUtils.conv2d_bn(tensor_x, 80, size=(1, 1), strides=(1, 1), padding='valid')  # 73 * 73 * 80
    tensor_x = KerasUtils.conv2d_bn(tensor_x, 192, size=(3, 3), strides=(1, 1), padding='valid')  # 71 * 71 * 192
    tensor_x = layers.MaxPooling2D((3, 3), strides=(2, 2))(tensor_x)  # 35 * 35 * 192

    # As in figure5 part1
    # 当前向量是 35*35*192
    branch1x1 = KerasUtils.conv2d_bn(tensor_x, 64, size=(1, 1))  # 35 * 35 * 64

    branch5x5 = KerasUtils.conv2d_bn(tensor_x, 48, size=(1, 1))  # 35 * 35 * 48
    branch5x5 = KerasUtils.conv2d_bn(branch5x5, 64, size=(5, 5))  # 35 * 35 * 64

    branch3x3 = KerasUtils.conv2d_bn(tensor_x, 64, size=(1, 1))  # 35 * 35 * 64
    branch3x3 = KerasUtils.conv2d_bn(branch3x3, 96, size=(3, 3))  # 35 * 35 * 96
    branch3x3 = KerasUtils.conv2d_bn(branch3x3, 96, size=(3, 3))  # 35 * 35 * 96

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(tensor_x)  # 35 * 35 * 192
    branch_pool = KerasUtils.conv2d_bn(branch_pool, 32, (1, 1))  # 35 * 35 * 32

    # 64 + 64 + 96 + 32 = 256
    # [n, h, w, c] => [0, 1, 2, 3]
    tensor_x = layers.concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=3)  # 35 * 35 * 256

    # As in figure5 part2
    # 35*35*256 => 35 * 35 * 288
    br1x1 = KerasUtils.conv2d_bn(tensor_x, 64, size=(1, 1))  # 35 * 35 * 64

    br5x5 = KerasUtils.conv2d_bn(tensor_x, 48, size=(1, 1))  # 35 * 35 * 48
    br5x5 = KerasUtils.conv2d_bn(br5x5, 64, size=(5, 5))  # 35 * 35 * 64

    br3x3 = KerasUtils.conv2d_bn(tensor_x, 64, size=(1, 1))
    br3x3 = KerasUtils.conv2d_bn(br3x3, 96, size=(3, 3))
    br3x3 = KerasUtils.conv2d_bn(br3x3, 96, size=(3, 3))  # 35 * 35 * 96

    br_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(tensor_x)
    br_pool = KerasUtils.conv2d_bn(br_pool, 64, size=(1, 1))  # 35 * 35 * 64

    # 64 + 64 + 96 + 64 => 288
    tensor_x = layers.concatenate([br1x1, br5x5, br3x3, br_pool], axis=3)

    # As in figure5 part3
    # 35 * 35 * 288 => 35 * 35 * 288
    br1x1 = KerasUtils.conv2d_bn(tensor_x, 64, size=(1, 1))  # 35 * 35 * 64

    br5x5 = KerasUtils.conv2d_bn(tensor_x, 48, size=(1, 1))  # 35 * 35 * 48
    br5x5 = KerasUtils.conv2d_bn(br5x5, 64, size=(5, 5))  # 35 * 35 * 64

    br3x3 = KerasUtils.conv2d_bn(tensor_x, 64, size=(1, 1))
    br3x3 = KerasUtils.conv2d_bn(br3x3, 96, size=(3, 3))
    br3x3 = KerasUtils.conv2d_bn(br3x3, 96, size=(3, 3))  # 35 * 35 * 96

    br_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(tensor_x)
    br_pool = KerasUtils.conv2d_bn(br_pool, 64, size=(1, 1))  # 35 * 35 * 64

    # 64 + 64 + 96 + 64 => 288
    tensor_x = layers.concatenate([br1x1, br5x5, br3x3, br_pool], axis=3)

    # As in figure6 part1 (17 * 17 * 768)
    # 35 * 35 * 288 => 17 * 17 * 768
    print("tensor_x:", tensor_x.shape)
    br3x3 = KerasUtils.conv2d_bn(tensor_x, 384, size=(3, 3), strides=(2, 2), padding="valid")  # 17 * 17 * 384

    br3x3_dbl = KerasUtils.conv2d_bn(tensor_x, 64, size=(1, 1))  # 35 * 35 * 64
    br3x3_dbl = KerasUtils.conv2d_bn(br3x3_dbl, 96, size=(3, 3))  # 35 * 35 * 94
    br3x3_dbl = KerasUtils.conv2d_bn(br3x3_dbl, 96, size=(3, 3), strides=(2, 2), padding="valid")  # 17 * 17 * 96

    br_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(tensor_x)  # 17 * 17 * 288

    # 384 + 96 + 288 = 768
    tensor_x = layers.concatenate([br3x3, br3x3_dbl, br_pool], axis=3)

    # As in figure6  (17 * 17 * 768)
    # Part2 17 * 17 * 768 => 17 * 17 * 768
    br1x1 = KerasUtils.conv2d_bn(tensor_x, 192, size=(1, 1))  # 17 * 17 * 192

    br7x7 = KerasUtils.conv2d_bn(tensor_x, 128, size=(1, 1))  # 17 * 17 * 128
    br7x7 = KerasUtils.conv2d_bn(br7x7, 128, size=(1, 7))  # 17 * 17 * 128
    br7x7 = KerasUtils.conv2d_bn(br7x7, 192, size=(7, 1))  # 17 * 17 * 192

    br7x7dbl = KerasUtils.conv2d_bn(tensor_x, 128, size=(1, 1))  # 17 * 17 * 128
    br7x7dbl = KerasUtils.conv2d_bn(br7x7dbl, 128, size=(7, 1))  # 17 * 17 * 128
    br7x7dbl = KerasUtils.conv2d_bn(br7x7dbl, 128, size=(1, 7))  # 17 * 17 * 128
    br7x7dbl = KerasUtils.conv2d_bn(br7x7dbl, 128, size=(7, 1))  # 17 * 17 * 128
    br7x7dbl = KerasUtils.conv2d_bn(br7x7dbl, 192, size=(1, 7))  # 17 * 17 * 192

    br_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(tensor_x)
    br_pool = KerasUtils.conv2d_bn(br_pool, 192, size=(1, 1))  # 17 * 17 * 192

    # Result: 17 * 17 * (192 + 192 + 192 + 192)
    tensor_x = layers.concatenate([br1x1, br7x7, br7x7dbl, br_pool], axis=3)

    # As in figure6  (17 * 17 * 768)
    # Part3 & Part4  17 * 17 * 768 => 17 * 17 * 768
    for i in range(2):
        br1x1 = KerasUtils.conv2d_bn(tensor_x, 192, size=(1, 1))  # 17 * 17 * 192

        br7x7 = KerasUtils.conv2d_bn(tensor_x, 160, size=(1, 1))  # 17 * 17 * 160
        br7x7 = KerasUtils.conv2d_bn(br7x7, 160, size=(1, 7))  # 17 * 17 * 160
        br7x7 = KerasUtils.conv2d_bn(br7x7, 192, size=(7, 1))  # 17 * 17 * 192

        br7x7dbl = KerasUtils.conv2d_bn(tensor_x, 160, size=(1, 1))  # 17 * 17 * 160
        br7x7dbl = KerasUtils.conv2d_bn(br7x7dbl, 160, size=(7, 1))  # 17 * 17 * 160
        br7x7dbl = KerasUtils.conv2d_bn(br7x7dbl, 160, size=(1, 7))  # 17 * 17 * 160
        br7x7dbl = KerasUtils.conv2d_bn(br7x7dbl, 160, size=(7, 1))  # 17 * 17 * 160
        br7x7dbl = KerasUtils.conv2d_bn(br7x7dbl, 192, size=(1, 7))  # 17 * 17 * 192

        br_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(tensor_x)
        br_pool = KerasUtils.conv2d_bn(br_pool, 192, size=(1, 1))  # 17 * 17 * 192

        tensor_x = layers.concatenate([br1x1, br7x7, br7x7dbl, br_pool], axis=3)

    # As in figure6  (17 * 17 * 768)
    # Part5  17 * 17 * 768 => 17 * 17 * 768
    br1x1 = KerasUtils.conv2d_bn(tensor_x, 192, size=(1, 1))  # 17 * 17 * 192

    br7x7 = KerasUtils.conv2d_bn(tensor_x, 192, size=(1, 1))
    br7x7 = KerasUtils.conv2d_bn(br7x7, 192, size=(1, 7))
    br7x7 = KerasUtils.conv2d_bn(br7x7, 192, size=(7, 1))  # 17 * 17 * 192

    br7x7dbl = KerasUtils.conv2d_bn(tensor_x, 192, size=(1, 1))  # 17 * 17 * 160
    br7x7dbl = KerasUtils.conv2d_bn(br7x7dbl, 192, size=(7, 1))  # 17 * 17 * 160
    br7x7dbl = KerasUtils.conv2d_bn(br7x7dbl, 192, size=(1, 7))  # 17 * 17 * 160
    br7x7dbl = KerasUtils.conv2d_bn(br7x7dbl, 192, size=(7, 1))  # 17 * 17 * 160
    br7x7dbl = KerasUtils.conv2d_bn(br7x7dbl, 192, size=(1, 7))  # 17 * 17 * 192

    br_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(tensor_x)
    br_pool = KerasUtils.conv2d_bn(br_pool, 192, size=(1, 1))  # 17 * 17 * 192
    # 192 + 192 +192 + 192 = 768
    tensor_x = layers.concatenate([br1x1, br7x7, br7x7dbl, br_pool], axis=3)

    # As in figure7  (8 * 8 * 1280)
    # Part1 17 * 17 * 768 ->  8 x 8 x 704
    br3x3 = KerasUtils.conv2d_bn(tensor_x, 192, size=(1, 1))  # 17 * 17 * 192
    br3x3 = KerasUtils.conv2d_bn(br3x3, 320, size=(3, 3), strides=(2, 2), padding="valid")  # 8 * 8 * 320

    br7x7x3 = KerasUtils.conv2d_bn(tensor_x, 192, size=(1, 1))  # 17 * 17 * 192
    br7x7x3 = KerasUtils.conv2d_bn(br7x7x3, 192, size=(1, 7))  # 17 * 17 * 192
    br7x7x3 = KerasUtils.conv2d_bn(br7x7x3, 192, size=(7, 1))  # 17 * 17 * 192
    br7x7x3 = KerasUtils.conv2d_bn(br7x7x3, 192, size=(3, 3), strides=(2, 2), padding="valid")  # 8 * 8 *192

    br_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(tensor_x)  # 8 * 8 * 768
    # 320 + 192 + 768 = 1280
    tensor_x = layers.concatenate([br3x3, br7x7x3, br_pool], axis=3)

    # As in figure7  (8 * 8 * 1280)
    # Part2 & Part3  8 * 8 * 1280 -> 8 * 8 * 2048
    for i in range(2):
        br1x1 = KerasUtils.conv2d_bn(tensor_x, 320, size=Size.x1_1)  # 8 * 8 * 320

        br3x3 = KerasUtils.conv2d_bn(tensor_x, 384, size=Size.x1_1)  # 8 * 8 * 384
        br3x3_1 = KerasUtils.conv2d_bn(br3x3, 384, size=Size.x1_3)  # 8 * 8 * 384
        br3x3_2 = KerasUtils.conv2d_bn(br3x3_1, 384, size=Size.x3_1)  # 8 * 8 * 384
        br3x3 = layers.concatenate([br3x3_1, br3x3_2], axis=3, name=f"mixed_f7_p{i}_b1")  # 8 * 8 * 768

        br3x3bdl = KerasUtils.conv2d_bn(tensor_x, 488, size=Size.x1_1)  # 8 * 8 * 488
        br3x3bdl = KerasUtils.conv2d_bn(br3x3bdl, 384, size=Size.x3_3)  # 8 * 8 * 384
        br3x3bdl_1 = KerasUtils.conv2d_bn(br3x3bdl, 384, size=Size.x1_3)  # 8 * 8 * 384
        br3x3bdl_2 = KerasUtils.conv2d_bn(br3x3bdl, 384, size=Size.x3_1)  # 8 * 8 * 384
        br3x3bdl = layers.concatenate([br3x3bdl_1, br3x3bdl_2], axis=3, name=f"mixed_f7_p{i}_b2")  # 8 * 8 * 768

        br_pool = layers.AveragePooling2D(Size.x3_3, strides=(1, 1), padding="same")(tensor_x)  # 8 * 8 * 1280
        br_pool = KerasUtils.conv2d_bn(br_pool, 192, Size.x1_1)  # 8 * 8 * 192

        # 320 + 768 + 768 + 192 = 2048
        tensor_x = layers.concatenate([br1x1, br3x3, br3x3bdl, br_pool], axis=3)

    tensor_x = layers.GlobalAvgPool2D(name="avg_pool")(tensor_x)
    tensor_x = layers.Dense(classes, activation="softmax")(tensor_x)

    return Model(img_tensor, tensor_x, name="inception_v3")


def preprocess_input(intput_tenser):
    intput_tenser /= 255.  # [0.0, 1.0]
    intput_tenser -= 0.5  # [-0.5, 0.5]
    intput_tenser *= 2.  # [-1.0, 1.0]
    return intput_tenser


if __name__ == '__main__':
    # padding  计算公式
    # output_height = (H - KH) / SH + 1
    # output_width = (W - KW) / SH + 1

    # P = (KH - 1) / 2
    # (H + 2P - KH) / SH + 1 =  H  =>  H + 2P - KH = (H -1) * SH =>  2P = (H -1) * SH +  KH - H
    # P = ((SH - 1) * H + (KH - SH)) / 2

    print("V3")
    m = model()
    m.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = m.predict(x)
