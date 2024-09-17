import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, DepthwiseConv2D, Input, \
    BatchNormalization, Activation, GlobalAveragePooling2D, Reshape, Dropout
from keras import Model
from keras import backend as k
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions


def relu6(x):
    return k.relu(x, max_value=6)


def conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(
        filters,
        kernel,
        padding='same',
        use_bias=False,
        strides=strides,
        name='conv1'
    )(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(relu6, name='conv1_relu')(x)
    return x


def depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1,
                         strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier,
                        strides=strides, use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(
        pointwise_conv_filters, (1, 1), padding='same', use_bias=False,
        strides=(1, 1), name='conv_pw_%d' % block_id
    )(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

    return x


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def MobileNet(input_shape=None, depth_multiplier=1, dropout=1e-3, classes=1000):
    if input_shape is None:
        input_shape = [224, 224, 3]

    img_input = Input(shape=input_shape)

    # 224,224,3 -> 112,112,32
    x = conv_block(img_input, 32, strides=(2, 2))

    # 112,112,32 -> 112,112,64
    x = depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # 112,112,64 -> 56,56,128
    x = depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)

    # 56,56,128 -> 56,56,128
    x = depthwise_conv_block(x, 128, block_id=3)

    # 56,56,128 -> 28,28,256
    x = depthwise_conv_block(x, 256, strides=(2, 2), block_id=4)

    # 28,28,256 -> 28,28,56
    x = depthwise_conv_block(x, 256, block_id=5)

    # 28,28,256 -> 14,14,512
    x = depthwise_conv_block(x, 512, strides=(2, 2), block_id=6)

    # 5次 14,14,512 -> 14,14,512
    for i in range(5):
        x = depthwise_conv_block(x, 512, block_id=6+i+1)

    # 14,14,512 -> 7,7,1024
    x = depthwise_conv_block(x, 1024, strides=(2, 2), block_id=12)

    # 7,7,1024 -> 7,7,1024
    x = depthwise_conv_block(x, 1024, block_id=13)

    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)

    # 在本例中全局平均池化层的输出已经为1,1,1024，所以这个Reshape的有无无伤大雅
    x = Reshape((1, 1, 1024), name='reshape_1')(x)

    x = Dropout(dropout, name='dropout')(x)

    # 用卷积代替全连接
    x = Conv2D(classes, (1, 1), name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)

    x = Reshape((classes,), name='reshape_2')(x)

    inputs = img_input

    return Model(inputs, x, name='mobilenet_1_0_224_tf')


if __name__ == '__main__':
    model = MobileNet()
    model.load_weights('./mobilenet_1_0_224_tf.h5')

    img = image.load_img('./elephant.jpg', target_size=(224, 224))

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    print('Input image shape: ', img.shape)

    preds = model.predict(img)

    print(np.argmax(preds))
    print('Predicted: ', decode_predictions(preds))
