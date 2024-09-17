from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import  np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import  backend as K
K.set_image_data_format('channels_last')

def generateArraysFromFile(lines, batchSize):

    n = len(lines)
    i = 0
    while True :
        Xtrain, Ytrain = [], []
        for b in range(batchSize): #获取一个batchSize大小的数据
            if i == 0 : np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            img = cv2.imread(r'.\data\image\train' + '/' + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            Xtrain.append(img)
            Ytrain.append(lines[i].split(';')[1])
            i = (i + 1) % n #读完一个周期后重新开始
        #处理图像
        Xtrain = utils.resizeImage(Xtrain, (224, 224))
        Xtrain = Xtrain.reshape(-1, 224, 224, 3)
        Ytrain = np_utils.to_categorical(np.array(Ytrain), num_classes=2)
        yield (Xtrain, Ytrain)

if __name__ == '__main__':

    #模型保存位置
    logDir = './log/'
    #打开数据集txt
    with open(r'.\data\dataset.txt', 'r') as f: lines = f.readlines()
    #打乱行
    np.random.seed(99610)
    np.random.shuffle(lines)
    np.random.seed(None)
    #90%用于训练，10%用于评估
    numVal = int(len(lines) * 0.1)
    numTrain = len(lines) - numVal
    #建立模型
    model = AlexNet()
    #3世代保存一次
    checkpointPeriod1 = ModelCheckpoint(logDir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                        monitor='acc',
                                        save_weights_only=False,
                                        save_best_only=True, period=3)
    #acc 三次不下降就下降学习率继续训练
    reduceLr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1)
    #当val_loss一直下降意味着模型基本训练完毕，可以停止
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    model.compile(loss = 'categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    #一次训练集大小
    batchSize = 128
    print("Train on {} samples, val on {} samples, with batch size {}.".format(numTrain, numVal, batchSize))
    #开始训练
    model.fit_generator(generateArraysFromFile(lines[:numTrain], batchSize),
                        steps_per_epoch=max(1, numTrain//batchSize),
                        validation_data=generateArraysFromFile(lines[numTrain:], batchSize),
                        validation_steps=max(1, numVal//batchSize),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpointPeriod1, reduceLr])
    model.save_weights(logDir + 'last.h5')
