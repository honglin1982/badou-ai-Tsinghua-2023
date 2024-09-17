import numpy as np

from keras.preprocessing import image
from keras.models import  Model
from keras.layers import  DepthwiseConv2D, Input, Activation, \
    Dropout, Reshape, BatchNormalization, GlobalAveragePooling2D, Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K

def convBlock(inputs, filters, kernel=(3, 3), strides=(1, 1)):

    x = Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides, name='conv1')(inputs)
    x = BatchNormalization(name='conv1Bn')(x)
    return Activation(relu6, name='conv1Relu')(x)

def depthWiseConvBlock(inputs, pointWiaseConvFilters, depthMultiplier=1, strides=(1, 1), blockId = 1):

    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depthMultiplier,
                        strides=strides, use_bias=False, name='convDw%d'%blockId)(inputs)
    x = BatchNormalization(name='convDw%dbn'%blockId)(x)
    x = Activation(relu6, name='convDw%drelu'%blockId)(x)

    #重点关注相关 Depthwise 有没有 附带 PointWise (一般都没有)
    x = Conv2D(pointWiaseConvFilters, (1, 1), padding='same', use_bias=False,
               strides=(1, 1), name='convPw%d' % blockId)(x)
    x = BatchNormalization(name='convPw%dbn'%blockId)(x)
    return Activation(relu6, name='convPw%drelu'%blockId)(x)

def relu6(x) : return K.relu(x, max_value=6)

def preprocessInput(x):

    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def MobileNet(inputShape=[224, 224, 3], depthMultiplier=1, dropout=1e-3, classes=1000):

    imgInput = Input(shape=inputShape)
    #224,224,3 -> 112,112,32
    x = convBlock(imgInput, 32, strides=(2, 2))
    # 112,112,32 -> 112,112,64
    x = depthWiseConvBlock(x, 64, depthMultiplier, blockId=1)
    # 112,112,64 -> 56,56,128
    x = depthWiseConvBlock(x, 128, depthMultiplier, strides=(2, 2), blockId=2)
    # 56,56,128 -> 56,56,128
    x = depthWiseConvBlock(x, 128, depthMultiplier, blockId=3)
    # 56,56,128 -> 28,28,256
    x = depthWiseConvBlock(x, 256, depthMultiplier, strides=(2, 2), blockId=4)
    # 28,28,256 -> 28,28,256
    x = depthWiseConvBlock(x, 256, depthMultiplier, blockId=5)
    # 28,28,256 -> 14,14,512
    x = depthWiseConvBlock(x, 512, depthMultiplier, strides=(2, 2), blockId=6)

    # 14,14,512 -> 14,14,512
    for i in range(7, 12):
        x = depthWiseConvBlock(x, 512, depthMultiplier, blockId=i)
    # 14,14,512 -> 7,7,1024
    x = depthWiseConvBlock(x, 1024, depthMultiplier, strides=(2, 2), blockId=12)
    x = depthWiseConvBlock(x, 1024, depthMultiplier, blockId=13)
    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='rehshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='convPreds')(x)
    x = Activation('softmax', name='actSoftmax')(x)
    x = Reshape((classes,), name= 'reshape_2')(x)

    inputs = imgInput
    model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    model.load_weights('mobilenet_1_0_224_tf.h5')
    return model

if __name__ == '__main__':

    model = MobileNet(inputShape=(224, 224, 3))
    img = image.load_img('elephant.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocessInput(x)

    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds, 1))