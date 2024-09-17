import numpy as np
from keras import  layers
from keras.layers import  Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import  Activation, BatchNormalization, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input



def identityBlock(inputTensor, kernelSize, filters, stage, block):

    filters1, filters2, filters3 = filters
    convNameBase = 'res' + str(stage) + block +"_branch"
    bnNameBase = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=convNameBase + '2a')(inputTensor)
    x = BatchNormalization(name=bnNameBase + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernelSize, padding='same', name=convNameBase+'2b')(x)
    x = BatchNormalization(name=bnNameBase +'2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=convNameBase + '2c')(x)
    x = BatchNormalization(name=bnNameBase + '2c')(x)
    x = layers.add([x, inputTensor])
    x = Activation('relu')(x)
    return x

def convBlock(inputTensor, kernelSize, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters
    convNameBase = 'res' + str(stage) + block +'_branch'
    bnNameBase = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=convNameBase + '2a')(inputTensor)
    x = BatchNormalization(name=bnNameBase+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernelSize, padding='same', name=convNameBase +'2b')(x)
    x = BatchNormalization(name=bnNameBase+'2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=convNameBase + '2c')(x)
    x = BatchNormalization(name=bnNameBase + '2c')(x)
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=convNameBase + '1')(inputTensor)
    shortcut = BatchNormalization(name=bnNameBase+'1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(inputShape=[224, 224, 3], classes=1000):

    imgInput = Input(shape=inputShape)
    x = ZeroPadding2D((3, 3))(imgInput)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = convBlock(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identityBlock(x, 3, [64, 64, 256], stage=2, block='b')
    x = identityBlock(x, 3, [64, 64, 256], stage=2, block='c')

    x = convBlock(x, 3, [128, 128, 512], stage= 3, block='a')
    x = identityBlock(x, 3, [128, 128, 512], stage=3, block='b')
    x = identityBlock(x, 3, [128, 128, 512], stage=3, block='c')
    x = identityBlock(x, 3, [128, 128, 512], stage=3, block='d')

    x = convBlock(x, 3, [256, 256, 1024], stage= 4, block='a')
    x = identityBlock(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identityBlock(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identityBlock(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identityBlock(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identityBlock(x, 3, [256, 256, 1024], stage=4, block='f')

    x = convBlock(x, 3, [512, 512, 2048], stage= 5, block='a')
    x = identityBlock(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identityBlock(x, 3, [512, 512, 2048], stage=5, block='c')
    x = AveragePooling2D((7,7), name='avg_poll')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)
    model = Model(imgInput, x, name='resnet50')
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    return model

if __name__ == '__main__':

    model = ResNet50()
    model.summary()
    imgPath = 'bike.jpg'
    img = image.load_img(imgPath, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))