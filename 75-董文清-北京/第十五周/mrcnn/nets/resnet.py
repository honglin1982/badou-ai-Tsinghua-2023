from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add

def identityBlock(inputTensor, kernelSize, filters, stage, block,
                  useBias = True, trainBn = True):

    nbFilter1, nbFilter2, nbFilter3 = filters
    convNameBase = 'res' + str(stage) + block + '_branch'
    bnNameBase = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nbFilter1, (1, 1), name = convNameBase + '2a', use_bias = useBias)(inputTensor)
    x = BatchNormalization(name = bnNameBase + '2a')(x, training = trainBn)
    x = Activation('relu')(x)

    x = Conv2D(nbFilter2, (kernelSize, kernelSize), padding = 'same',
               name = convNameBase + '2b', use_bias = useBias)(x)
    x = BatchNormalization(name = bnNameBase + '2b')(x, training = trainBn)
    x = Activation('relu')(x)

    x = Conv2D(nbFilter3, (1, 1), name = convNameBase + '2c', use_bias = useBias)(x)
    x = BatchNormalization(name = bnNameBase + '2c')(x, training = trainBn)
    x = Add()([x, inputTensor])
    x = Activation('relu', name = 'res' + str(stage) + block + '_out')(x)
    return x

def convBlock(inputTensor, kernelSize, filters, stage, block, strides = (2, 2),
              useBias = True, trainBn = True) :
    nbFilter1, nbFilter2, nbFilter3 = filters
    convNameBase = 'res' + str(stage) + block + '_branch'
    bnNameBase = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nbFilter1, (1, 1), strides = strides,
                name = convNameBase + '2a', use_bias = useBias)(inputTensor)
    x = BatchNormalization(name = bnNameBase + '2a')(x, training = trainBn)
    x = Activation('relu')(x)

    x = Conv2D(nbFilter2, (kernelSize, kernelSize), padding = 'same',
               name = convNameBase + '2b', use_bias = useBias)(x)
    x = BatchNormalization(name = bnNameBase + '2b')(x, training = trainBn)
    x = Activation('relu')(x)

    x = Conv2D(nbFilter3, (1, 1), name = convNameBase + '2c', use_bias = useBias)(x)
    x = BatchNormalization(name = bnNameBase + '2c')(x, training = trainBn)

    shortcut = Conv2D(nbFilter3, (1, 1), strides = strides, name = convNameBase + '1',
                      use_bias = useBias)(inputTensor)
    shortcut = BatchNormalization(name = bnNameBase + '1')(shortcut, training = trainBn)

    x = Add()([x, shortcut])
    x = Activation('relu', name = 'res' + str(stage) + block + '_out')(x)
    return x

def getResNet(inputImage, stage5 = False, trainBn = True):

    #stage 1
    x = ZeroPadding2D((3, 3))(inputImage)
    x = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', use_bias = True)(x)
    x = BatchNormalization(name = 'bn_conv1')(x, training = trainBn)
    x = Activation('relu')(x)
    # Height/4,Width/4,64
    C1 = x = MaxPooling2D((3, 3), strides = (2, 2), padding = 'same')(x)

    #stage 2
    x = convBlock(x, 3, [64, 64, 256], stage = 2, block = 'a', strides = (1, 1), trainBn = trainBn)
    x = identityBlock(x, 3, [64, 64, 256], stage = 2, block = 'b', trainBn = trainBn)
    # Height/4,Width/4,256
    C2 = x = identityBlock(x, 3, [64, 64, 256], stage = 2, block = 'c', trainBn = trainBn)

    #stage3
    x = convBlock(x, 3, [128, 128, 512], stage = 3, block = 'a', trainBn = trainBn)
    x = identityBlock(x, 3, [128, 128, 512], stage = 3, block = 'b', trainBn = trainBn)
    x = identityBlock(x, 3, [128, 128, 512], stage = 3, block = 'c', trainBn = trainBn)
    # Height/8,Width/8,512
    C3 = x = identityBlock(x, 3, [128, 128, 512], stage = 3, block = 'd', trainBn = trainBn)

    #stage4
    x = convBlock(x, 3, [256, 256, 1024], stage = 4, block = 'a', trainBn = trainBn)
    blockCount = 22
    for i in range(blockCount):
        x = identityBlock(x, 3, [256, 256, 1024], stage = 4, block = chr(98 + i), trainBn = trainBn)
    # Height/16,Width/16,1024
    C4 = x
    #stage 5
    if stage5:
        x = convBlock(x, 3, [512, 512, 2048], stage = 5, block = 'a', trainBn = trainBn)
        x = identityBlock(x, 3, [512, 512, 2048], stage = 5, block = 'b', trainBn = trainBn)
        # Height/32,Width/32,2048
        C5 = x = identityBlock(x, 3, [512, 512, 2048], stage = 5, block = 'c', trainBn = trainBn)
    else : C5 = None
    return [C1, C2, C3, C4, C5]