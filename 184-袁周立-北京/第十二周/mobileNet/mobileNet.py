import keras
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np


def _conv_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    return x


def _depthwise_separable_conv_block(inputs, filters, strides=(1, 1)):
    x = layers.DepthwiseConv2D((3, 3),
                               strides=strides,
                               padding='same',
                               depth_multiplier=1,
                               use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    x = layers.Conv2D(filters, (1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    return x


def mobile_net(input_shape=(224, 224, 3), class_num=1000):
    img_input = layers.Input(input_shape)

    x = _conv_block(img_input, 32, (3, 3), strides=(2, 2))  # (224, 224, 3) => (112, 112, 32)

    x = _depthwise_separable_conv_block(x, 64)                      # => (112, 112, 64)
    x = _depthwise_separable_conv_block(x, 128, strides=(2, 2))     # => (56, 56, 128)
    x = _depthwise_separable_conv_block(x, 128)                     # => (56, 56, 128)
    x = _depthwise_separable_conv_block(x, 256, strides=(2, 2))     # => (28, 28, 256)
    x = _depthwise_separable_conv_block(x, 256)                     # => (28, 28, 256)
    x = _depthwise_separable_conv_block(x, 512, strides=(2, 2))     # => (14, 14, 512)

    x = _depthwise_separable_conv_block(x, 512)                     # => (14, 14, 512)
    x = _depthwise_separable_conv_block(x, 512)                     # => (14, 14, 512)
    x = _depthwise_separable_conv_block(x, 512)                     # => (14, 14, 512)
    x = _depthwise_separable_conv_block(x, 512)                     # => (14, 14, 512)
    x = _depthwise_separable_conv_block(x, 512)                     # => (14, 14, 512)

    x = _depthwise_separable_conv_block(x, 1024, strides=(2, 2))    # => (7, 7, 1024)
    x = _depthwise_separable_conv_block(x, 1024)                    # => (7, 7, 1024)

    x = layers.GlobalAveragePooling2D()(x)
    # keras内置的mobileNet中，有这几个reshape，所以要加上保持一致，不然形状对应不上，它的weights加载不了
    x = layers.Reshape((1, 1, 1024))(x)                             # => (1, 1, 1024)
    x = layers.Conv2D(class_num, (1, 1), activation='softmax')(x)   # => (1, 1, 1000)
    x = layers.Reshape((class_num, ))(x)                            # => (1000, )

    model = keras.Model(img_input, x, name='mobile_net')
    model.summary()
    return model


def main():
    model = mobile_net(input_shape=(224, 224, 3))
    model.load_weights('./mobilenet_1_0_224_tf.h5')

    img = image.load_img('elephant.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x, mode='tf')

    predict = model.predict(x)
    print(decode_predictions(predict))


if __name__ == '__main__':
    main()
