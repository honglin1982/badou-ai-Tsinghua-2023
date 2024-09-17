from .resnet import ResNet50, classifierLayers
from keras.layers import Conv2D, Input, TimeDistributed, Flatten, Dense, Reshape
from keras.models import Model
from .RoiPoolingConv import RoiPoolingConv

def getRpn(baseLayers, numAnchors):

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal',
               name='rpn_conv1')(baseLayers)
    xClass = Conv2D(numAnchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                    name='rpn_out_class')(x)
    xRegr = Conv2D(numAnchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                    name='rpn_out_regress')(x)
    xClass = Reshape((-1, 1), name='classification')(xClass)
    xRegr = Reshape((-1, 4), name='regression')(xRegr)
    return [xClass, xRegr, baseLayers]

def getClassifier(baseLayers, inputRois, numRois, nbClasses=21, trainable=True):

    poolingRegions = 14
    inputShape = (numRois, 14, 14, 1024)
    outRoiPool = RoiPoolingConv(poolingRegions, numRois)([baseLayers, inputRois])
    out = classifierLayers(outRoiPool, inputShape=inputShape, trainable=trainable)
    out = TimeDistributed(Flatten())(out)
    outClass = TimeDistributed(Dense(nbClasses, activation='softmax', kernel_initializer='zero'),
                               name='dense_class_{}'.format(nbClasses))(out)
    outRegr = TimeDistributed(Dense(4 * (nbClasses - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nbClasses))(out)
    return [outClass, outRegr]

def getModel(config, numClasses):

    inputs = Input(shape=(None, None, 3))
    roiInput = Input(shape=(None, 4))
    baseLaysers = ResNet50(inputs)

    numAnchors = len(config.anchorBoxScales) * len(config.anchorBoxRatios)
    rpn = getRpn(baseLaysers, numAnchors)
    modelRpn = Model(inputs, rpn[:2])

    classifier = getClassifier(baseLaysers, roiInput, config.numRois, nbClasses=numClasses, trainable=True)
    modelClassifier = Model([inputs, roiInput], classifier)
    modelAll = Model([inputs, roiInput], rpn[:2] + classifier)
    return modelRpn, modelClassifier, modelAll

def getPredictModel(config, numClasses):

    inputs = Input(shape=(None, None, 3))
    roiInput = Input(shape=(None, 4))
    featureMapInput = Input(shape=(None, None, 1024))

    baseLayers = ResNet50(inputs)
    numAnchors = len(config.anchorBoxScales) * len(config.anchorBoxRatios)
    rpn = getRpn(baseLayers, numAnchors)
    modelRpn = Model(inputs, rpn)

    classifier = getClassifier(featureMapInput, roiInput, config.numRois, nbClasses=numClasses, trainable=True)
    modelClassifierOnly = Model([featureMapInput, roiInput], classifier)
    return modelRpn, modelClassifierOnly

