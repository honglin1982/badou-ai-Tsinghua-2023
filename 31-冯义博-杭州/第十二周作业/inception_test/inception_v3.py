from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Input, MaxPooling2D, AveragePooling2D, concatenate, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import decode_predictions
import numpy as np


"""
构建统一卷积层
"""


def conv2d_bn(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(kernel_size=kernel_size,
               strides=strides,
               filters=filters,
               padding=padding,
               use_bias=False,
               name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def inceptionV3(input_shape=(299, 299, 3), classes=1000):
    input_img = Input(shape=input_shape)

    # inputs=(299, 299, 3) strides=2 kernel_size=3*3
    x = conv2d_bn(input_img, 32, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, padding='valid')
    x = conv2d_bn(x, 64)
    # 池化 3*3 strides=2
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, kernel_size=(1, 1), padding='valid')
    x = conv2d_bn(x, 192, padding='valid')
    # 池化 3*3 strides=2
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # block1 part1
    branch1x1 = conv2d_bn(x, 64, kernel_size=(1, 1))

    branch5x5 = conv2d_bn(x, 48, kernel_size=(1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, kernel_size=(5, 5))

    branch3x3 = conv2d_bn(x, 64, kernel_size=(1, 1))
    branch3x3 = conv2d_bn(branch3x3, 96)
    branch3x3 = conv2d_bn(branch3x3, 96)

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)

    # 64+64+96+32  nhwc axis c维度相加
    x = concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=3, name="mixed0")

    # block1 part2
    branch1x1 = conv2d_bn(x, 64, kernel_size=(1, 1))

    branch5x5 = conv2d_bn(x, 48, kernel_size=(1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, kernel_size=(5, 5))

    branch3x3 = conv2d_bn(x, 64, kernel_size=(1, 1))
    branch3x3 = conv2d_bn(branch3x3, 96)
    branch3x3 = conv2d_bn(branch3x3, 96)

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    x = concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=3, name='mixed1')

    # block1 part3
    branch1x1 = conv2d_bn(x, 64, kernel_size=(1, 1))

    branch5x5 = conv2d_bn(x, 48, kernel_size=(1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, kernel_size=(5, 5))

    branch3x3 = conv2d_bn(x, 64, kernel_size=(1, 1))
    branch3x3 = conv2d_bn(branch3x3, 96)
    branch3x3 = conv2d_bn(branch3x3, 96)

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    x = concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=3, name='mixed2')

    # block2 part1
    branch3x3 = conv2d_bn(x, 384, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, kernel_size=(1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

    # block2 part2
    branch1x1 = conv2d_bn(x, 192, kernel_size=(1, 1))

    branch7x7 = conv2d_bn(x, 128, kernel_size=(1, 1))
    branch7x7 = conv2d_bn(branch7x7, 128, kernel_size=(1, 7))
    branch7x7 = conv2d_bn(branch7x7, 192, kernel_size=(7, 1))

    branch7x7dbl = conv2d_bn(x, 128, kernel_size=(1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, kernel_size=(7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, kernel_size=(1, 7))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, kernel_size=(7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(1, 7))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, kernel_size=(1, 1))
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed4')

    # block2 part3 & part4
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

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                        axis=3,
                        name='mixed' + str(5 + i))

    # block2 part5
    branch1x1 = conv2d_bn(x, 192, kernel_size=(1, 1))

    branch7x7 = conv2d_bn(x, 192, kernel_size=(1, 1))
    branch7x7 = conv2d_bn(branch7x7, 192, kernel_size=(1, 7))
    branch7x7 = conv2d_bn(branch7x7, 192, kernel_size=(7, 1))

    branch7x7dbl = conv2d_bn(x, 192, kernel_size=(1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(1, 7))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(1, 7))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, kernel_size=(1, 1))
    x = concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed7')

    # block3 part1
    branch3x3 = conv2d_bn(x, 192, kernel_size=(1, 1))
    branch3x3 = conv2d_bn(branch3x3, 320, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, kernel_size=(1, 1))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, kernel_size=(1, 7))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, kernel_size=(7, 1))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    # block3 part2 & part3
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, kernel_size=(1, 1))

        branch3x3 = conv2d_bn(x, 384, kernel_size=(1, 1))
        branch3x3_1 = conv2d_bn(branch3x3, 384, kernel_size=(1, 3))
        branch3x3_2 = conv2d_bn(branch3x3, 384, kernel_size=(3, 1))
        branch3x3 = concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, kernel_size=(1, 1))
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, kernel_size=(1, 3))
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, kernel_size=(3, 1))
        branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, kernel_size=(1, 1))
        x = concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed' + str(9 + i))

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = input_img
    model = Model(inputs, x, name='inception_v3')
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    return model



def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == "__main__":
    model = inceptionV3()


    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))


