import cv2
import utils
import  numpy as np
from keras.layers import  Conv2D, Input, MaxPool2D, Flatten, Dense, Permute
from keras.models import Model
from keras.layers.advanced_activations import PReLU
#-----------------------------#
#   粗略获取人脸框
#   输出bbox位置和是否有人脸
#-----------------------------#
def createPnet(weightPath):

    input = Input(shape = [None, None, 3])
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name = 'conv1')(input)
    x = PReLU(shared_axes=[1,2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    #无激活函数，线性
    bboxRegress = Conv2D(4, (1, 1), name = 'conv4-2')(x)
    model = Model([input], [classifier, bboxRegress])
    model.load_weights(weightPath, by_name=True)
    return model
# mtcnn 第二段 精修框
def createRnet(weightPath):

    input = Input(shape=[24, 24, 3])
    # 24,24,3 -> 11,11,28
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 11,11,28 -> 4,4,48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    # 3,3,64 -> 64,3,3
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    # 576 -> 128
    x = Dense(128, name = 'conv4')(x)
    x = PReLU(name = 'prelu4')(x)
    # 128 -> 2 128 -> 4
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bboxRegress = Dense(4, name='conv5-2')(x)
    model = Model([input], [classifier, bboxRegress])
    model.load_weights(weightPath, by_name=True)
    return model

def createOnet(weightPath):

    input = Input(shape = [48, 48, 3])
    # 48,48,3 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # 23,23,32 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    # 8,8,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)
    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)
    # 3,3,128 -> 128,12,12
    x = Permute((3, 2, 1))(x)
    # 1152 -> 256
    x = Flatten()(x)
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)
    # 鉴别
    # 256 -> 2 256 -> 4 256 -> 10
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    bboxRegress = Dense(4,name='conv6-2')(x)
    landmarkRegress = Dense(10, name='conv6-3')(x)

    model = Model([input], [classifier, bboxRegress, landmarkRegress])
    model.load_weights(weightPath, by_name=True)
    return model

class mtcnn():
    def __init__(self):
        self.Pnet = createPnet('model_data/pnet.h5')
        self.Rnet = createRnet('model_data/rnet.h5')
        self.Onet = createOnet('model_data/onet.h5')

    def detectFace(self, img, threshold):
        # -----------------------------#
        #   归一化，加快收敛速度
        #   把[0,255]映射到(-1,1)
        # -----------------------------#
        copyImg = (img.copy() - 127.5) / 127.5
        originH, originW, _ = copyImg.shape
        #   计算原始输入图像
        #   每一次缩放的比例
        scales = utils.calculateScales(img)
        out = []
        #   粗略计算人脸框
        #   pnet部分
        for scale in scales:
            hs = int(originH * scale)
            ws = int(originW * scale)
            scaleImg = cv2.resize(copyImg, (ws, hs))
            inputs = scaleImg.reshape(1, *scaleImg.shape)
            # 图像金字塔中的每张图片分别传入Pnet得到output
            output = self.Pnet.predict(inputs)
            out.append(output)

        imageNum = len(scales)
        rectangles = []
        for i in range(imageNum):
            #有人脸的概率
            clsProb = out[i][0][0][:,:,1]
            #其对应框的位置
            roi = out[i][1][0]

            #取出每个缩放后图片的长宽
            outH, outW = clsProb.shape
            outSide = max(outH, outW)
            print(clsProb.shape)
            #解码过程
            rectangle = utils.detectFace12(clsProb, roi, outSide, 1 / scales[i],
                                            originW, originH, threshold[0])
            rectangles.extend(rectangle)
        #进行非极大值抑制
        rectangles = utils.NMS(rectangles, 0.7)
        if len(rectangles) == 0: return rectangles

        # 稍微精确计算人脸框 Rnet
        predict24Batch = []
        for rectangle in rectangles:
            cropImg = copyImg[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scaleImg = cv2.resize(cropImg, (24, 24))
            predict24Batch.append(scaleImg)
        predict24Batch = np.array(predict24Batch)
        out = self.Rnet.predict(predict24Batch)

        clsProb = out[0]
        clsProb = np.array(clsProb)
        roiProb = out[1]
        roiProb = np.array(roiProb)
        rectangles = utils.filterFace24net(clsProb, roiProb, rectangles, originW, originH, threshold[1])
        if len(rectangles) == 0: return rectangles

        #计算人脸框 Onet部分
        predictBatch = []
        for rectangle in rectangles:
            cropImg = copyImg[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scaleImg = cv2.resize(cropImg, (48, 48))
            predictBatch.append(scaleImg)

        predictBatch = np.array(predictBatch)
        output = self.Onet.predict(predictBatch)
        clsProb = output[0]
        roiProb = output[1]
        ptsProb = output[2]
        rectangles = utils.filterFace48net(clsProb, roiProb, ptsProb, rectangles, originW, originH, threshold[2])

        return rectangles