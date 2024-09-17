# ------------------------------------------------------- #
# 卷积输出shape公式（默认情况）(valid)
# 卷积输出高度 h = round((输入高度 - 卷积核高度 + 1) / 步长)
# 卷积输出宽度 w = round((输入宽度 - 卷积核宽度 + 1) / 步长)
# (same)
# 输出高度 h = ceil(输入高度 / 步长)
# 输出宽度 w = ceil(输入宽度 / 步长)
# ------------------------------------------------------- #
# ------------------------------------------------------- #
# 池化输出shape公式（默认情况）
# output_h = floor((h - pool_h) / stride_h) + 1
# output_w = floor((w - pool_w) / stride_w) + 1
# ------------------------------------------------------- #
import numpy as np
from keras import layers
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization,\
    Activation, Input, AveragePooling2D, GlobalAveragePooling2D
from keras import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions


def conv2d_bn(x, filters, num_row, num_col, strides=(1, 1), padding='same', name=None):
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
    else:
        conv_name = None
        bn_name = None
    x = Conv2D(
        filters,
        (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name
    )(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def InceptionV3(input_shape=None, classes=1000):
    if input_shape is None:
        input_shape = [299, 299, 3]
    # 构建inceptionV3网络
    img_input = Input(shape=input_shape)

    # 第一部分，简单串联部分
    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')  # 149, 149, 32
    x = conv2d_bn(x, 32, 3, 3, padding='valid')  # 147, 147, 32
    x = conv2d_bn(x, 64, 3, 3)  # 147, 147, 64
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 73, 73, 64

    x = conv2d_bn(x, 80, 1, 1)  # 73, 73, 64
    x = conv2d_bn(x, 192, 3, 3, padding='valid')  # 71, 71, 192
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 35, 35, 192

    # 后面inception部分
    # -------------------------- #
    #        block1 35x35
    # -------------------------- #
    # block1 part1
    # 35, 35, 192 -> 35, 35, 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)  # 35, 35, 64

    branch5x5 = conv2d_bn(x, 48, 1, 1)  # 35, 35, 48
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)  # 35, 35, 64

    # 两个3x3代替5x5，v2第一种模式
    branch3x3db1 = conv2d_bn(x, 64, 1, 1)  # 35, 35, 64
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)  # 35, 35, 96
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)  # 35, 35, 96

    # 先池化再1x1
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)  # 35, 35, 192
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)  # 35, 35, 32

    # 64+64+96+32=256
    x = layers.concatenate([branch1x1, branch5x5, branch3x3db1, branch_pool], axis=3, name='mixed0')

    # block1 part2
    # 35, 35, 256 -> 35, 35, 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)  # 35, 35, 64

    branch5x5 = conv2d_bn(x, 48, 1, 1)  # 35, 35, 48
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)  # 35, 35, 64

    # 两个3x3代替5x5，v2第一种模式
    branch3x3db1 = conv2d_bn(x, 64, 1, 1)  # 35, 35, 64
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)  # 35, 35, 96
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)  # 35, 35, 96

    # 先池化再1x1
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)  # 35, 35, 256
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)  # 35, 35, 64

    # 64+64+96+64=288
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3db1, branch_pool],
        axis=3,
        name='mixed1'
    )

    # block1 part3
    # 35, 35, 288 -> 35, 35, 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)  # 35, 35, 64

    branch5x5 = conv2d_bn(x, 48, 1, 1)  # 35, 35, 48
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)  # 35, 35, 64

    # 两个3x3代替5x5，v2第一种模式
    branch3x3db1 = conv2d_bn(x, 64, 1, 1)  # 35, 35, 64
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)  # 35, 35, 96
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)  # 35, 35, 96

    # 先池化再1x1
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)  # 35, 35, 256
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)  # 35, 35, 64

    # 64+64+96+64=288
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3db1, branch_pool],
        axis=3,
        name='mixed2'
    )

    # -------------------------- #
    #        block2 17x17
    # -------------------------- #
    # block2 part1
    # 35, 35, 288 -> 17, 17, 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')  # 17, 17, 384

    # 两个3x3代替5x5
    branch3x3db1 = conv2d_bn(x, 64, 1, 1)  # 35, 35, 64
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)  # 35, 35, 96
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3, strides=(2, 2), padding='valid')  # 17, 17, 96

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 17, 17, 288

    # 384+96+288=768
    x = layers.concatenate([branch3x3, branch3x3db1, branch_pool], axis=3, name='mixed3')

    # block2 part2
    # 17, 17, 768 -> 17, 17, 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)  # 17, 17, 192

    # 用1xn+nx1的代替nxn，v2第二种模式
    branch7x7 = conv2d_bn(x, 128, 1, 1)  # 17, 17, 128
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)  # 17, 17, 128
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)  # 17, 17, 192

    # 用两个7x7代替9x9，并且每个7x7用上面的方法替代，顺序调换了一下
    branch7x7db1 = conv2d_bn(x, 128, 1, 1)  # 17, 17, 128
    branch7x7db1 = conv2d_bn(branch7x7db1, 128, 7, 1)  # 17, 17, 128
    branch7x7db1 = conv2d_bn(branch7x7db1, 128, 1, 7)  # 17, 17, 128
    branch7x7db1 = conv2d_bn(branch7x7db1, 128, 7, 1)  # 17, 17, 128
    branch7x7db1 = conv2d_bn(branch7x7db1, 192, 1, 7)  # 17, 17, 192

    # 使用的是平均池化
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)  # 17, 17, 768
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)  # 17, 17, 192

    # 192+192+192+192 = 768
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7db1, branch_pool],
        axis=3,
        name='mixed4'
    )

    # block2 part3 and part4
    # 17, 17, 768 -> 17, 17, 768 -> 17, 17, 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7db1 = conv2d_bn(x, 160, 1, 1)
        branch7x7db1 = conv2d_bn(branch7x7db1, 160, 7, 1)
        branch7x7db1 = conv2d_bn(branch7x7db1, 160, 1, 7)
        branch7x7db1 = conv2d_bn(branch7x7db1, 160, 7, 1)
        branch7x7db1 = conv2d_bn(branch7x7db1, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7db1, branch_pool],
            axis=3,
            name='mixed' + str(5 + i)
        )

    # block2 part5
    # 17, 17, 768 -> 17, 17, 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7db1 = conv2d_bn(x, 192, 1, 1)
    branch7x7db1 = conv2d_bn(branch7x7db1, 192, 7, 1)
    branch7x7db1 = conv2d_bn(branch7x7db1, 192, 1, 7)
    branch7x7db1 = conv2d_bn(branch7x7db1, 192, 7, 1)
    branch7x7db1 = conv2d_bn(branch7x7db1, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7db1, branch_pool],
        axis=3,
        name='mixed7'
    )

    # ------------------------- #
    #        block3 8x8
    # ------------------------- #
    # block3 part1
    # 17, 17, 768 -> 8, 8, 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)  # 17, 17, 192
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')  # 8, 8, 320

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    # 加上一个3x3的卷积
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')  # 8, 8, 192

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 8, 8, 768

    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=3,
        name='mixed8'
    )

    # block3 part2 and part3
    # 8, 8, 1280 -> 8, 8, 2048 -> 8, 8, 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)  # 8, 8, 320

        # 用1x3和3x1代替3x3，并联
        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i)
        )  # 8, 8, 768

        # 用两个3x3代替5x5，第二个3x3用上面的并联结构代替
        branch3x3db1 = conv2d_bn(x, 448, 1, 1)
        branch3x3db1 = conv2d_bn(branch3x3db1, 384, 3, 3)  # 8, 8, 384
        branch3x3db1_1 = conv2d_bn(branch3x3db1, 384, 1, 3)
        branch3x3db1_2 = conv2d_bn(branch3x3db1, 384, 3, 1)
        branch3x3db1 = layers.concatenate(
            [branch3x3db1_1, branch3x3db1_2],
            axis=3,
        )  # 8, 8, 768

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)  # 8, 8, 192

        # 320+768+768+192=2048
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3db1, branch_pool],
            axis=3,
            name='mixed' + str(9 + i)
        )

    # 平均池化后全连接
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # 将输入 inputs 和输出 x 构建成一个模型，并将模型命名为 'inception_V3'
    m = Model(inputs, x, name='inception_V3')
    return m


def preprocess_input(x):
    # 自定义归一化函数
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = InceptionV3()
    model.load_weights('./inception_v3_weights_tf_dim_ordering_tf_kernels.h5')

    # 读取图片
    img = image.load_img('./elephant.jpg', target_size=(299, 299))
    # 将图片转化为numpy数组、再转化为张量(n,h,w,c)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # 归一化
    img = preprocess_input(img)

    pre = model.predict(img)
    print('Predicted: ', decode_predictions(pre))
