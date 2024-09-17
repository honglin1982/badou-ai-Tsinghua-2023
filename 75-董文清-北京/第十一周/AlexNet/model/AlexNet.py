from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

def AlexNet(inputShape=(224, 224, 3), outputShape=2):

    model = Sequential()
    #添加步长4*4 大小11的卷积核，输出特征曾为96层 输出shape(55, 55, 96) 为了加快收敛 以下filter减半
    model.add(Conv2D(filters=48, kernel_size=(11, 11), strides=(4, 4),
                      padding='valid', input_shape=inputShape, activation='relu'))
    model.add(BatchNormalization()) #进行批次归一化操作
    #使用步长为2的最大池化层进行池化, 输出shape为(27, 27, 96)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    #使用步长 1 * 1， 大小为 5 的卷积核进行卷积，输出特征层256, shape(27, 27, 256);
    model.add(Conv2D(filters=128, kernel_size=(5,5), strides=(1, 1),
                     padding='same', activation='relu'))
    model.add(BatchNormalization())
    #使用步长为2的最大池化层进行池化 输出shape(13, 13, 256)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    #使用步长1 * 1，大小为3的卷积进行卷积，输出特征层384，输出shape1为(13, 13, 384)
    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    #使用步长1 * 1，大小为3的卷积进行卷积，输出特征层384，输出shape1为(13, 13, 256)
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu'))
    #使用步长为2的最大池化层池化， 输出shape(6, 6, 256)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    #加入全连接层 最后输出1000类 这里改为两类
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(outputShape, activation='softmax'))
    return model
