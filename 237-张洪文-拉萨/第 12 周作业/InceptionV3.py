# 用于在 Python 2.x 中编写与 Python 3.x 更兼容的代码
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image


# 层集合：二维卷积、BN正态化、激活函数
def con2d_bn(x,filters,kernel_size,strides=(1,1),padding="same",name=None):
    # 根据传入的 name 参数, 设置conv和bn层名
    if name is not None:
        conv_name = name + "_conv"
        bn_name = name + "_bn"
    else:
        conv_name = None
        bn_name = None

    # 各层具体设置
    x = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=False,  # 不使用偏置项
                      name=conv_name)(x)
    x = layers.BatchNormalization(scale=False, name=bn_name)(x)  # scale=False 不进行缩放
    x = layers.Activation(activation="relu", name=name)(x)
    return x  # 返回模型对象


# InceptionV3 网络模型
def inceptionV3(input_shape=(299,299,3), classes=1000):
    # 定义输入层
    img_input = layers.Input(shape=input_shape)

    x = con2d_bn(img_input, 32, kernel_size=(3, 3), strides=(2, 2), padding="valid")
    x = con2d_bn(x, 32, (3, 3), padding="valid")
    x = con2d_bn(x, 64, (3, 3))
    x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    x = con2d_bn(x, 80, (1, 1), padding="valid")
    x = con2d_bn(x, 192, (3, 3), padding="valid")
    x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    # --------------------------------#
    #   Block1 35x35
    # --------------------------------#
    # Block1 part1: 35 x 35 x 192 -> 35 x 35 x 256
    branch_1x1 = con2d_bn(x, 64, kernel_size=(1, 1))

    branch_5x5 = con2d_bn(x, 48, (1, 1))
    branch_5x5 = con2d_bn(branch_5x5, 64, (5, 5))

    branch_3x3_db1 = con2d_bn(x, 64, (1, 1))
    branch_3x3_db1 = con2d_bn(branch_3x3_db1, 96, (3, 3))
    branch_3x3_db1 = con2d_bn(branch_3x3_db1, 96, (3, 3))

    branch_pool = layers.AveragePooling2D((3,3), strides=(1,1), padding="same")(x)  # 平均池化层
    branch_pool = con2d_bn(branch_pool, 32, kernel_size=(1, 1))

    # shape=nhwc,所以axis=3，即沿 c 进行合并，64+64+96+32 = 256
    x = layers.concatenate(
        [branch_1x1, branch_5x5, branch_3x3_db1, branch_pool],
        axis=3, name="mixed0")

    # Block1 part2: 35 x 35 x 256 -> 35 x 35 x 288
    branch_1x1 = con2d_bn(x, 64, (1, 1))

    branch_5x5 = con2d_bn(x, 48, (1, 1))
    branch_5x5 = con2d_bn(branch_5x5, 64, (5, 5))

    branch_3x3_db1 = con2d_bn(x, 64, (1, 1))
    branch_3x3_db1 = con2d_bn(branch_3x3_db1, 96, (3, 3))
    branch_3x3_db1 = con2d_bn(branch_3x3_db1, 96, (3, 3))

    branch_pool = layers.AveragePooling2D((3,3), (1,1), padding="same")(x)
    branch_pool = con2d_bn(branch_pool, 64, (1, 1))

    x = layers.concatenate(
        [branch_1x1, branch_5x5, branch_3x3_db1, branch_pool],
        axis=3, name="mixed1")

    # Block1 part3: 35 x 35 x 288 -> 35 x 35 x 288
    branch_1x1 = con2d_bn(x, 64, (1, 1))

    branch_5x5 = con2d_bn(x, 48, (1, 1))
    branch_5x5 = con2d_bn(branch_5x5, 64, (5, 5))

    branch_3x3_db1 = con2d_bn(x, 64, (1, 1))
    branch_3x3_db1 = con2d_bn(branch_3x3_db1, 96, (3, 3))
    branch_3x3_db1 = con2d_bn(branch_3x3_db1, 96, (3, 3))

    branch_pool = layers.AveragePooling2D((3, 3), (1, 1), padding="same")(x)
    branch_pool = con2d_bn(branch_pool, 64, (1, 1))

    x = layers.concatenate(
        [branch_1x1, branch_5x5, branch_3x3_db1, branch_pool],
        axis=3, name="mixed2")


    # --------------------------------#
    #   Block2 17x17
    # --------------------------------#
    # Block2 part1: 35 x 35 x 288 -> 17 x 17 x 768
    branch_3x3 = con2d_bn(x, 384, (3, 3), strides=(2, 2), padding="valid")

    branch_3x3_db1 = con2d_bn(x, 64, (1, 1))
    branch_3x3_db1 = con2d_bn(branch_3x3_db1, 96, (3, 3))
    branch_3x3_db1 = con2d_bn(branch_3x3_db1, 96, (3, 3), strides=(2, 2), padding="valid")

    branch_pool = layers.MaxPooling2D((3, 3),strides=(2, 2))(x)
    x = layers.concatenate(
        [branch_3x3, branch_3x3_db1, branch_pool],
        axis=3, name="mixed3")

    # Block2 part2: 17 x 17 x 768 -> 17 x 17 x 768
    branch_3x3 = con2d_bn(x, 192, (1, 1))

    branch_7x7 = con2d_bn(x, 128, (1, 1))
    branch_7x7 = con2d_bn(branch_7x7, 128, (1, 7))
    branch_7x7 = con2d_bn(branch_7x7, 192, (7, 1))

    branch_7x7_db1 = con2d_bn(x, 128, (1, 1))
    branch_7x7_db1 = con2d_bn(branch_7x7_db1, 128, (7, 1))
    branch_7x7_db1 = con2d_bn(branch_7x7_db1, 128, (1, 7))
    branch_7x7_db1 = con2d_bn(branch_7x7_db1, 128, (7, 1))
    branch_7x7_db1 = con2d_bn(branch_7x7_db1, 192, (1, 7))

    branch_pool = layers.AveragePooling2D((3, 3),strides=(1, 1),padding="same")(x)
    branch_pool = con2d_bn(branch_pool, 192, (1, 1))
    x = layers.concatenate(
        [branch_3x3, branch_7x7, branch_7x7_db1, branch_pool],
        axis=3, name="mixed4")

    # Block2 part3 and part4:
    # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):
        branch_1x1 = con2d_bn(x, 192, (1, 1))

        branch_7x7 = con2d_bn(x, 160, (1, 1))
        branch_7x7 = con2d_bn(branch_7x7, 160, (1, 7))
        branch_7x7 = con2d_bn(branch_7x7, 192, (7, 1))

        branch_7x7_db1 = con2d_bn(x, 160, (1, 1))
        branch_7x7_db1 = con2d_bn(branch_7x7_db1, 160, (7, 1))
        branch_7x7_db1 = con2d_bn(branch_7x7_db1, 160, (1, 7))
        branch_7x7_db1 = con2d_bn(branch_7x7_db1, 160, (7, 1))
        branch_7x7_db1 = con2d_bn(branch_7x7_db1, 192, (1, 7))

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = con2d_bn(branch_pool, 192, (1, 1))
        x = layers.concatenate(
            [branch_1x1, branch_7x7, branch_7x7_db1, branch_pool],
            axis=3, name="mixed" + str(5+i))

    # Block2 part5: 17 x 17 x 768 -> 17 x 17 x 768
    branch_1x1 = con2d_bn(x, 192, (1, 1))

    branch_7x7 = con2d_bn(x, 192, (1, 1))
    branch_7x7 = con2d_bn(branch_7x7, 192, (1, 7))
    branch_7x7 = con2d_bn(branch_7x7, 192, (7, 1))

    branch_7x7_db1 = con2d_bn(x, 192, (1, 1))
    branch_7x7_db1 = con2d_bn(branch_7x7_db1, 192, (7, 1))
    branch_7x7_db1 = con2d_bn(branch_7x7_db1, 192, (1, 7))
    branch_7x7_db1 = con2d_bn(branch_7x7_db1, 192, (7, 1))
    branch_7x7_db1 = con2d_bn(branch_7x7_db1, 192, (1, 7))

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = con2d_bn(branch_pool, 192, (1, 1))
    x = layers.concatenate(
        [branch_1x1, branch_7x7, branch_7x7_db1, branch_pool],
        axis=3, name="mixed7")

    # --------------------------------#
    #   Block3 8x8
    # --------------------------------#
    # Block3 part1: 17 x 17 x 768 -> 8 x 8 x 1280
    branch_3x3 = con2d_bn(x, 192, (1, 1))
    branch_3x3 = con2d_bn(branch_3x3, 320, (3, 3), strides=(2, 2), padding="valid")

    branch_7x7x3 = con2d_bn(x, 192, (1, 1))
    branch_7x7x3 = con2d_bn(branch_7x7x3, 192, (1, 7))
    branch_7x7x3 = con2d_bn(branch_7x7x3, 192, (7, 1))
    branch_7x7x3 = con2d_bn(branch_7x7x3, 192, (3, 3), (2, 2), padding="valid")

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch_3x3, branch_7x7x3, branch_pool], axis=3, name="mixed8")

    # Block3 part2 and part3: 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch_1x1 = con2d_bn(x, 320, (1, 1))

        branch_3x3 = con2d_bn(x, 384, (1, 1))
        branch_3x3_1 = con2d_bn(branch_3x3, 384, (1, 3))
        branch_3x3_2 = con2d_bn(branch_3x3, 384, (3, 1))
        branch_3x3 = layers.concatenate(
            [branch_3x3_1, branch_3x3_2], axis=3, name="mixed9_"+str(i))

        branch_3x3_db1 = con2d_bn(x, 448, (1, 1))
        branch_3x3_db1 = con2d_bn(branch_3x3_db1, 384, (3, 3))
        branch_3x3_db1_1 = con2d_bn(branch_3x3_db1, 384, (1, 3))
        branch_3x3_db1_2 = con2d_bn(branch_3x3_db1, 384, (3, 1))
        branch_3x3_db1 = layers.concatenate(
            [branch_3x3_db1_1, branch_3x3_db1_2], axis=3)

        branch_pool = layers.AveragePooling2D((3,3), strides=(1,1), padding="same")(x)
        branch_pool = con2d_bn(branch_pool, 192, (1, 1))
        x = layers.concatenate(
            [branch_1x1, branch_3x3, branch_3x3_db1,branch_pool],
            axis=3, name="mixed"+str(9+i))

    # 全局平均池化后进行全连接
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dense(units=classes, activation="softmax", name="predictions")(x)

    inputs = img_input
    model = models.Model(inputs, x, name="inception_v3")

    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = inceptionV3()
    # 加载权重
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    img_path = "./elephant.jpg"
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)  # 预处理

    preds = model.predict(x)  # 推理
    print("Predicted: ", decode_predictions(preds))
