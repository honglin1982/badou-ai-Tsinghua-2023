#resnet50
from keras import layers
from keras.layers import Input
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, TimeDistributed, Add
from keras.layers import Activation

import keras.backend as K
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers

class BatchNormalization(Layer):

    #初始化各类参数
    def __init__(self, epsilon=1e-3, axis=-1, weights=None,
                 betaInit='zero', gammaInit='one',
                 gammaRegularizer=None, betaRegularizer=None, **kwargs):

        self.supports_masking = True
        self.betaInit = initializers.get(betaInit)
        self.gammaInit = initializers.get(gammaInit)
        self.epsilon = epsilon
        self.axis = axis
        self.gammaRegularizer = regularizers.get(gammaRegularizer)
        self.betaRegularizer = regularizers.get(betaRegularizer)
        self.initialWeights = weights
        super(BatchNormalization, self).__init__(**kwargs)
    #构建输入
    def build(self, inputShape):

        self.inputSpec = [InputSpec(shape=inputShape)]
        shape = (inputShape[self.axis],)
        self.gamma = self.add_weight(shape=shape, initializer=self.gammaInit,
                                     regularizer=self.gammaRegularizer,
                                     name='{}_gamma'.format(self.name),
                                     trainable=False)
        self.beta = self.add_weight(shape=shape, initializer=self.betaInit,
                                    regularizer=self.betaRegularizer,
                                    name='{}_beta'.format(self.name),
                                    trainable=False)
        self.runningMean = self.add_weight(shape=shape, initializer='zero',
                                           name='{}_runningMean'.format(self.name),
                                           trainable=False)
        self.runningStd = self.add_weight(shape=shape, initializer='one',
                                          name='{}_runningStd'.format(self.name),
                                          trainable=False)
        if self.initialWeights is not None:
            self.set_weights(self.initialWeights)
            del self.initialWeights
        self.built = True

    def call(self, x, mask=None):

        assert self.built, 'Layer doesn\'t build, Layer must be built before being called'
        inputShape = K.int_shape(x)

        reductionAxes = list(range(len(inputShape)))
        del reductionAxes[self.axis]
        broadcastShape = [1] * len(inputShape)
        broadcastShape[self.axis] = inputShape[self.axis]

        if sorted(reductionAxes) == range(K.ndim(x))[:-1]:

            xNormed = K.batch_normalization(x, self.runningMean, self.runningStd,
                                            self.beta, self.gamma, epsilon=self.epsilon)
        else:
            #维度不同，需要广播
            broadcastRunningMean = K.reshape(self.runningMean, broadcastShape)
            broadcastRunningStd = K.reshape(self.runningStd, broadcastShape)
            broadcastBeta = K.reshape(self.beta, broadcastShape)
            broadcastGamma = K.reshape(self.gamma, broadcastShape)
            xNormed = K.batch_normalization(x, broadcastRunningMean, broadcastRunningStd,
                                            broadcastBeta, broadcastGamma,
                                            epsilon=self.epsilon)
        return xNormed

    def get_config(self):
        config = { 'epsilon': self.epsilon,
                   'axis' : self.axis,
                   'gammaRegularizer': self.gammaRegularizer.get_config()
                    if self.gammaRegularizer else None,
                   'betaRegularizer': self.betaRegularizer.get_config()
                    if self.betaRegularizer else None}
        baseConfig = super(BatchNormalization, self).get_config()
        return dict(list(baseConfig.items()) + list(config.items()))

def identityBlock(inputTensor, kernelSize, filters, stage, block):

    filters1, filters2, filters3 = filters
    convNameBase = 'res' + str(stage) + block + '_branch'
    bnNameBase = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=convNameBase+'2a')(inputTensor)
    x = BatchNormalization(name=bnNameBase+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernelSize, padding='same', name=convNameBase+'2b')(x)
    x = BatchNormalization(name=bnNameBase+'2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=convNameBase+'2c')(x)
    x = BatchNormalization(name=bnNameBase+'2c')(x)

    x = layers.add([x, inputTensor])
    x = Activation('relu')(x)
    return x

def convBlock(inputTensor, kernelSize, filters, stage, block, strides=(2,2)):

    filters1, filters2, filters3 = filters
    convNameBase = 'res' + str(stage) + block + '_branch'
    bnNameBase = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=convNameBase + '2a')(inputTensor)
    x = BatchNormalization(name=bnNameBase + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernelSize, padding='same', name=convNameBase + '2b')(x)
    x = BatchNormalization(name=bnNameBase + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=convNameBase + '2c')(x)
    x = BatchNormalization(name=bnNameBase + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name = convNameBase + '1')(inputTensor)
    shortcut = BatchNormalization(name=bnNameBase + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(inputs):

    imgInput = inputs
    x = ZeroPadding2D((3, 3))(imgInput)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = convBlock(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    for i in range(1, 3):
      x = identityBlock(x, 3, [64, 64, 256], stage=2, block=chr(97 + i))

    x = convBlock(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, 4):
      x = identityBlock(x, 3, [128, 128, 512], stage=3, block=chr(97 + i))

    x = convBlock(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, 6):
        x = identityBlock(x, 3, [256, 256, 1024], stage=4, block=chr(97 + i))

    return x

def identityBlockTd(inputTensor, kernelSize, filters, stage, block, trainable=True):

    nbFilters1, nbFilters2, nbFilters3 = filters
    bnAxis = 3 if K.image_data_format() == 'channels_last' else 1
    convNameBase = 'res' + str(stage) + block + '_branch'
    bnNameBase = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nbFilters1, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=convNameBase+'2a')(inputTensor)
    x = TimeDistributed(BatchNormalization(axis=bnAxis), name=bnNameBase + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nbFilters2, (kernelSize, kernelSize), trainable=trainable,
                               kernel_initializer='normal', padding='same'),
                        name=convNameBase + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bnAxis), name=bnNameBase + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nbFilters3, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=convNameBase + '2c')(x)
    x = TimeDistributed(BatchNormalization(axis=bnAxis), name=bnNameBase + '2c')(x)

    x = Add()([x, inputTensor])
    x = Activation('relu')(x)
    return x

def convBlockTd(inputTensor, kernelSize, filters, stage, block, inputShape, strides=(2, 2), trainable=True):

    nbFilters1, nbFilters2, nbFilters3 = filters
    bnAxis = 3 if K.image_data_format() == 'channels_last' else 1
    convNameBase = 'res' + str(stage) + block + '_branch'
    bnNameBase = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nbFilters1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
                        input_shape=inputShape, name=convNameBase + '2a')(inputTensor)
    x = TimeDistributed(BatchNormalization(axis=bnAxis), name=bnNameBase + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nbFilters2, (kernelSize, kernelSize), padding='same', trainable=trainable,
                               kernel_initializer='normal'), name = convNameBase + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bnAxis), name=bnNameBase + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nbFilters3, (1, 1), kernel_initializer='normal'),
                        name=convNameBase + '2c', trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(axis=bnAxis), name=bnNameBase + '2c')(x)

    shortcut = TimeDistributed(Conv2D(nbFilters3, (1, 1), strides=strides, trainable=trainable,
                                      kernel_initializer='normal'), name=convNameBase + '1')(inputTensor)
    shortcut = TimeDistributed(BatchNormalization(axis=bnAxis), name=bnNameBase + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def classifierLayers(x, inputShape, trainable=False):

    x = convBlockTd(x, 3, [512, 512, 2048], stage=5, block='a',
                    inputShape=inputShape, strides=(2, 2), trainable=trainable)
    for i in range(1, 3):
        x = identityBlockTd(x, 3, [512, 512, 2048], stage=5, block=chr(97 + i), trainable=trainable)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avgPool')(x)

    return x

if __name__ == "__main__":
    inputs = Input(shape=(600, 600, 3))
    model = ResNet50(inputs)
    model.summary()
