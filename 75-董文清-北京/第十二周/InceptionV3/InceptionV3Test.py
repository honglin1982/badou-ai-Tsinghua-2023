import numpy as np
from keras.models import Model
from keras import  layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import  image

def conv2dBn(x, filters, numRow, numCol, strides=(1,1), padding='same', name = None):

    if name is not None:
        bnName, convName = name +'_bn', name +'_conv'
    else:
        bnName, convName = None, None

    x = Conv2D(filters, (numRow, numCol), strides=strides, padding=padding,
               use_bias=False, name=convName)(x)
    x = BatchNormalization(scale=False, name=bnName)(x)
    x = Activation('relu', name=name)(x)
    return x

def InceptionV3(inputShape=[299, 299, 3], classes=1000):

    imgInput = Input(shape=inputShape)

    #卷积部分
    x = conv2dBn(imgInput, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2dBn(x, 32, 3, 3, padding='valid')
    x = conv2dBn(x, 64, 3, 3)
    #池化
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    #卷积部分
    x = conv2dBn(x, 80, 1, 1, padding='valid')
    x = conv2dBn(x, 192, 3, 3, padding='valid')
    #池化
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    #Inception模块组 Block1 35 * 35 part1
    # 35 x 35 x 192 -> 35 x 35 x 256
    branch1x1 = conv2dBn(x, 64, 1, 1)

    branch5x5 = conv2dBn(x, 48, 1, 1)
    branch5x5 = conv2dBn(branch5x5, 64 , 5, 5)

    branch3x3dbl = conv2dBn(x, 64, 1, 1)
    #下两行代码等价于 5 * 5卷积 64 * 3
    branch3x3dbl = conv2dBn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2dBn(branch3x3dbl, 96, 3, 3)

    branchPool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branchPool = conv2dBn(branchPool, 32, 1, 1)
    #卷积数 64 + 64 + 96 + 32 = 256 nhwc~(0,1,2,3)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branchPool], axis=3, name='mixed0')

    #Block1 part2, 3
    #35 x 35 x 256 -> 35 x 35 x 288(64 + 64 + 96 + 64) -> 35 x 35 x 288
    for i in range(1, 3):
        branch1x1 = conv2dBn(x, 64, 1, 1)

        branch5x5 = conv2dBn(x, 48, 1, 1)
        branch5x5 = conv2dBn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2dBn(x, 64, 1, 1)
        branch3x3dbl = conv2dBn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2dBn(branch3x3dbl, 96, 3, 3)

        branchPool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branchPool = conv2dBn(branchPool, 64, 1, 1)
        x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branchPool], axis=3, name='mixed'+str(i))


    #Block2 17 * 17
    #Block part1 35 x 35 x 288 -> 17 x 17 x 768
    branch3x3 = conv2dBn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2dBn(x, 64, 1, 1)
    branch3x3dbl = conv2dBn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2dBn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branchPool = MaxPooling2D((3, 3), strides=(2, 2))(x) #288
    x = layers.concatenate([branch3x3, branch3x3dbl, branchPool], axis=3, name='mixed3')

    #Block2 part2 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2dBn(x, 192, 1, 1)

    branch7x7 = conv2dBn(x, 128, 1, 1)
    branch7x7 = conv2dBn(branch7x7, 128, 1, 7)
    branch7x7 = conv2dBn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2dBn(x, 128, 1, 1)
    branch7x7dbl = conv2dBn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2dBn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2dBn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2dBn(branch7x7dbl, 192, 1, 7)

    branchPool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branchPool = conv2dBn(branchPool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branchPool], axis=3, name='mixed4')

    #Block2 part3、4 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(5, 7):

        branch1x1 = conv2dBn(x, 192, 1, 1)

        branch7x7 = conv2dBn(x, 160, 1, 1)
        branch7x7 = conv2dBn(branch7x7, 160, 1, 7)
        branch7x7 = conv2dBn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2dBn(x, 160, 1, 1)
        branch7x7dbl = conv2dBn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2dBn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2dBn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2dBn(branch7x7dbl, 192, 1, 7)

        branchPool = AveragePooling2D((3, 3), strides=(1,1), padding='same')(x)
        branchPool = conv2dBn(branchPool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branchPool], axis=3, name='mixed' + str(i))

    #Block part5  17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2dBn(x, 192, 1, 1)

    branch7x7 = conv2dBn(x, 192, 1, 1)
    branch7x7 = conv2dBn(branch7x7, 192, 1, 7)
    branch7x7 = conv2dBn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2dBn(x, 192, 1, 1)
    branch7x7dbl = conv2dBn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2dBn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2dBn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2dBn(branch7x7dbl, 192, 1, 7)

    branchPool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branchPool = conv2dBn(branchPool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branchPool], axis=3, name='mixed7')

    # Block 8 x 8  part1 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv2dBn(x, 192, 1, 1)
    branch3x3 = conv2dBn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2dBn(x, 192, 1, 1)
    branch7x7x3 = conv2dBn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2dBn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2dBn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branchPool = MaxPooling2D((3, 3), strides=(2, 2))(x) #768
    x = layers.concatenate([branch3x3, branch7x7x3, branchPool], axis=3, name='mixed8')

    #Block3 part2,3  8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(9, 11):
        branch1x1 = conv2dBn(x, 320, 1, 1)

        branch3x3 = conv2dBn(x, 384, 1, 1)
        branch3x3_1 = conv2dBn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2dBn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixedBlock3_' + str(i))

        branch3x3dbl = conv2dBn(x, 448, 1, 1)
        branch3x3dbl = conv2dBn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2dBn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2dBn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3) #384 * 2

        branchPool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branchPool = conv2dBn(branchPool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branchPool], axis=3, name='mixed'+str(i))
    #池化 + FC
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = imgInput
    model = Model(inputs, x, name='inception_v3')
    return model

def preprocessInput(x):

    x /= 255. #归一化
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__' :

    model = InceptionV3()
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    imgPath = 'elephant.jpg'
    img = image.load_img(imgPath, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocessInput(x)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))