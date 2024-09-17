from utils.anchors import getAnchors
from keras import backend as K
from keras.applications.imagenet_utils import  preprocess_input
import keras
import tensorflow as tf
import numpy as np
from random import shuffle
import random
from PIL import Image
from keras.objectives import categorical_crossentropy
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def rand(a=0, b=1): return np.random.rand() * (b - a) + a

def clsLoss(ratio=3):

    def _clsLoss(yTrue, yPred):
        # y_true [batch_size, num_anchor, num_classes+1]
        # y_pred [batch_size, num_anchor, num_classes]
        labels = yTrue
        anchorState = yTrue[:, :, -1] # 0->背景，1->存在目标
        classification = yPred

        #寻找存在目标的先验框
        indicesForObject = tf.where(keras.backend.equal(anchorState, 1))
        labelsForObject =  tf.gather_nd(labels, indicesForObject)
        classificationForObject = tf.gather_nd(classification, indicesForObject)
        clsLossForObject = keras.backend.binary_crossentropy(labelsForObject, classificationForObject)

        #找出实际背景的先验框
        indicesForBack = tf.where(keras.backend.equal(anchorState, 0))
        labelsForBack  = tf.gather_nd(labels, indicesForBack)
        classificationForBack = tf.gather_nd(classification, indicesForBack)

        #计算每个先验框应有的权重
        clsLossForBack = keras.backend.binary_crossentropy(labelsForBack, classificationForBack)
        #标准化（正、负样本数）
        normalizerPos = tf.where(keras.backend.equal(anchorState, 1))
        normalizerPos = keras.backend.cast(keras.backend.shape(normalizerPos)[0], keras.backend.floatx())
        normalizerPos = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizerPos)

        normalizerNeg = tf.where(keras.backend.equal(anchorState, 0))
        normalizerNeg = keras.backend.cast(keras.backend.shape(normalizerNeg)[0], keras.backend.floatx())
        normalizerNeg = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizerNeg)

        #获取loss
        clsLossForObject = keras.backend.sum(clsLossForObject) / normalizerPos
        clsLossForBack = ratio * keras.backend.sum(clsLossForBack) / normalizerNeg
        loss = clsLossForObject + clsLossForBack
        return loss
    return _clsLoss

def smoothL1(sigma=1.0):

    sigmaSquared = sigma ** 2
    def _smoothL1(yTrue, yPred):
        # y_true [batch_size, num_anchor, 4+1]
        # y_pred [batch_size, num_anchor, 4]
        regresion = yPred
        regresionTarget = yTrue[:, : ,:-1]
        anchorState = yTrue[:, :, -1]

        #找正样本
        indices = tf.where(keras.backend.equal(anchorState, 1))
        regression = tf.gather_nd(regresion, indices)
        regressionTarget = tf.gather_nd(regresionTarget, indices)

        #计算  smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regressionDiff = regression - regressionTarget
        regressionDiff = keras.backend.abs(regressionDiff)
        regressionLoss = tf.where(keras.backend.less(regressionDiff, 1.0 / sigmaSquared),
                                  0.5 * sigmaSquared * keras.backend.pow(regressionDiff, 2),
                                  regressionDiff - 0.5 / sigmaSquared)
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss = keras.backend.sum(regressionLoss) / normalizer
        return loss
    return _smoothL1

def classLossRegr(numClasses):

    epsilon = 1e-4
    def classLossRegrFixedNum(yTrue, yPred):
        x = yTrue[:, :, 4 * numClasses] - yPred
        xAbs = K.abs(x)
        xBool = K.cast(K.less_equal(xAbs, 1.0), 'float32')
        loss = 4 * K.sum(yTrue[:, :, : 4 * numClasses] * (xBool * (0.5 * x * x) + (1 - xBool) * (xAbs - 0.5)))\
               / K.sum(epsilon + yTrue[:, :, : 4 * numClasses])
        return loss
    return classLossRegrFixedNum

def classLossCls(yTrue, yPred): return K.mean(categorical_crossentropy(yTrue[0, :, :], yPred[0, :, :]))

def getNewImgSize(width, height, imgMinSide=600):

    if width <= height:
        f = float(imgMinSide) / width
        resizedHeight = int(f * height)
        resizedWidth = int(imgMinSide)
    else:
        f = float(imgMinSide) / height
        resizedWidth = int(f * width)
        resizedHeight = int(imgMinSide)
    return resizedWidth, resizedHeight

def getImgOutputLength(width, height):
    def getOutputlength(inputLength):
        #inputLength += 6
        filterSizes = [7, 3, 1, 1]
        padding = [3, 1, 0, 0]
        stride = 2
        # input_length = (input_length - filter_size + stride) // stride
        for i in range(4) : inputLength = (inputLength + 2 * padding[i] - filterSizes[i]) // stride + 1
        return inputLength
    return getOutputlength(width), getOutputlength(height)

class Generator(object):

    def __init__(self, bboxUtil, trainLines, numClasses, solid, solidShape=[600, 600]):

            self.bboxUtil = bboxUtil
            self.trainLines = trainLines
            self.trainBatches = len(trainLines)
            self.numClasses = numClasses
            self.solid = solid
            self.solidShape = solidShape

    def getRandomData(self, annotationLine, random=True, jitter=.1, hue=.1, sat=1.1, val=1.1, procImg=True):

        '''实时数据增强的预处理'''
        line = annotationLine.split()
        image = Image.open(line[0])
        iw, ih = image.size
        if self.solid:
            w, h = self.solidShape
        else:
            w, h = getNewImgSize(iw, ih)
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        #resize image
        newAr = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(.9, 1,1)
        if newAr < 1 :
            nh = int(scale * h)
            nw = int(nh * newAr)
        else:
            nw = int(scale * w)
            nh = int(nw / newAr)
        image = image.resize((nw, nh), Image.BICUBIC) #双线性三次插值 保证图像缩放不丢失太多信息

        #place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        newImage = Image.new('RGB', (w, h), (128, 128, 128))
        newImage.paste((image, (dx, dy)))

        #flip image or not 翻转图片
        flip = rand() < .5
        image = image.transpose(Image.FLIP_LEFT_RIGHT) if flip else image

        #distort image 图像畸变
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] - 0
        imageData = hsv_to_rgb(x) * 255

        #correct boxes 修正
        boxData = np.zeros((len(box), 5))
        if len(box) > 0 :
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip : box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            boxW = box[:, 2] - box[:, 0]
            boxH = box[:, 3] - box[:, 1]
            box = box[np.logical_and(boxW > 1, boxH > 1)] #去掉无效的box
            boxData[:len(box)] = box
        if len(box) == 0 : return imageData, []
        if (boxData[:,:4] > 0).any() : return imageData,
        else: return imageData, []

    def generate(self):

        while True:
            shuffle(self.trainLines)
            lines = self.trainLines
            for annotationLine in lines:
                img, y = self.getRandomData(annotationLine)
                height, width = np.shape(img)[:2]
                if len(y) == 0: continue
                boxes = np.array(y[:,:4], dtype=np.float32)
                boxes[:,0] = boxes[:,0] / width
                boxes[:,1] = boxes[:,1] / height
                boxes[:,2] = boxes[:,2] / width
                boxes[:,3] = boxes[:,3] / height

                boxHeights = boxes[:,3] - boxes[:,1]
                boxWidths = boxes[:,2] - boxes[:,0]
                if (boxHeights <= 0).any() or (boxWidths <= 0).any(): continue

                y[:,:4] = boxes[:,:4]
                anchors = getAnchors(getImgOutputLength(width, height), width, height)

                #计算真实框对应的先验框以及该先验框的预测结果
                assignment = self.bboxUtil.assign_boxes(y, anchors)
                numRegions = 256
                classification = assignment[:,4]
                regression = assignment[:,:]

                maskPos = classification[:] > 0
                numPos = len(classification[maskPos])
                if numPos > numRegions / 2 :
                    valLocs = random.sample(range(numPos), int(numPos, numRegions / 2))
                    classification[maskPos][valLocs] = -1
                    regression[maskPos][valLocs, -1] = -1

                maskNeg = classification[:] == 0
                numNeg = len(classification[maskNeg])
                if len(classification[maskNeg]) + numPos > numRegions :
                    valLocs = random.sample(range(numNeg), int(numNeg - numPos))
                    classification[maskNeg][valLocs] = -1

                classification = np.reshape(classification,[-1, 1])
                regression = np.reshape(regression, [-1, 5])

                tmpInp = np.array(img)
                tmTargets = [np.expand_dims(np.array(classification, dtype=np.float32), 0),
                             np.expand_dims(np.array(regression, dtype=np.float32), 0)]
                yield preprocess_input(np.expand_dims(tmpInp, 0)), tmTargets, np.expand_dims(y, 0)