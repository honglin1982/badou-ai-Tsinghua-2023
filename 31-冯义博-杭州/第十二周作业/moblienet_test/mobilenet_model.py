from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Input, Activation, BatchNormalization, GlobalAveragePooling2D, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import decode_predictions

import tensorflow.keras.backend as K

import numpy as np


def relu6(x):
    return K.relu(x, max_value=6)


def conv_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = Conv2D(kernel_size=kernel_size,
               filters=filters,
               strides=strides,
               padding='same',
               use_bias=False,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(relu6, name='conv1_relu')(x)
    return x


def depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def mobilenet(input_shape=(224, 224, 3), classes=1000, depth_multiplier=1, dropout=1e-3):
    img_input = Input(shape=input_shape)

    # 224,224,3 -> 112,112,32
    x = conv_block(img_input, 32, strides=(2, 2))

    # 112,112,32 -> 112,112,64
    x = depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # 112,112,64 -> 56,56,128
    x = depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)
    # 56,56,128 -> 56,56,128
    x = depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 56,56,128 -> 28,28,256
    x = depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)

    # 28,28,256 -> 28,28,256
    x = depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 28,28,256 -> 14,14,512
    x = depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)

    # 14,14,512 -> 14,14,512
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024
    x = depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    inputs = img_input
    model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    model.load_weights('mobilenet_1_0_224_tf.h5')
    return model




def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x



if __name__ == "__main__":
    model = mobilenet(input_shape=(224, 224, 3))

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds,1))  # 只显示top1
