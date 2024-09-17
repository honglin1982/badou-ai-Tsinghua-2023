from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, TimeDistributed, Add, \
    Activation, Flatten, BatchNormalization, AveragePooling2D
from keras import layers
import keras.backend as k


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding="same", name=conv_name_base + '2b')(x)
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

    x = Conv2D(filters2, kernel_size, padding="same", name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    short_cut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    short_cut = BatchNormalization(name=bn_name_base + '1')(short_cut)

    x = layers.add([x, short_cut])
    x = Activation('relu')(x)

    return x


def ResNet50(inputs):
    # 若输入shape=(600, 600, 3)

    # 600x600x3 -> 300x300x64
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    # 300x300x64 -> 150x150x64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # 150x150x64 -> 150x150x256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # 150x150x256 -> 75x75x512
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # 75x75x512 -> 38x38x1024
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # 特征提取最终输出(-1, 38, 38, 1024)
    return x


def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
    nb_filters1, nb_filters2, nb_filters3 = filters
    # 每个通道归一化
    if k.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filters1, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filters2, kernel_size, trainable=trainable, kernel_initializer='normal',
                               padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filters3, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape,
                  strides=(2, 2), trainable=True):
    nb_filters1, nb_filters2, nb_filters3 = filters

    if k.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = f'bn{str(stage)}{block}_branch'

    x = TimeDistributed(Conv2D(nb_filters1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
                        input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(
        Conv2D(nb_filters2, kernel_size, padding='same',
               trainable=trainable, kernel_initializer='normal'),
        name=conv_name_base + '2b'
    )(x)
    x = TimeDistributed(
        BatchNormalization(axis=bn_axis),
        name=bn_name_base + '2b'
    )(x)
    x = Activation('relu')(x)

    x = TimeDistributed(
        Conv2D(
            nb_filters3, (1, 1), kernel_initializer='normal'
        ),
        name=conv_name_base + '2c',
        trainable=trainable
    )(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(
        Conv2D(
            nb_filters3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'
        ),
        name=conv_name_base + '1'
    )(input_tensor)
    shortcut = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def classifier_layers(x, input_shape, trainable=False):
    x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape,
                      strides=(2, 2), trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)

    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_poo')(x)

    return x
















