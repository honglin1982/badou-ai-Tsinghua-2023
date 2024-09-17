import os
import random
import config
import colorsys
import numpy as np
import tensorflow as tf
from model.yolo3Model import yolo

class yoloPredictor:

    def __init__(self, objThreshold, nmsThreshold, classesFile, anchorsFile):
        # 初始化函数 objThreshold 目标检测为物体的阈值
        self.objThreshold = objThreshold
        self.nmsThreshold = nmsThreshold
        #预读取
        self.classesPath = classesFile
        self.anchorsPath = anchorsFile
        #读取种类名称
        self.classesNames = self._getClass()
        #读取先验框
        self.anchors = self._getAnchors()
        #画框
        hsvTuples = [(x / len(self.classesNames), 1., 1.) for x in range(len(self.classesNames))]
        #颜色格式转换
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsvTuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(11032)
        random.shuffle(self.colors)
        random.seed(None)

    def _getClass(self):

        classesPath = os.path.expanduser(self.classesPath)
        with open(classesPath) as f: classNames = f.readlines()
        classNames = [c.strip() for c in classNames]
        return classNames

    def _getAnchors(self):

        anchorsPath = os.path.expanduser(self.anchorsPath)
        with open(anchorsPath) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors
    #对三个特征层进行解码，之后进行排序和nms
    def boxesAndScores(self, feats, anchors, classesNum, inputShape, imageShape):
        """
        Introduction
        ------------
            将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数
        Parameters
        ----------
            feats: yolo输出的feature map
            anchors: anchor的位置
            classesNum: 类别数目
            inputShape: 输入大小
            imageShape: 图片大小
        Returns
        -------
            boxes: 物体框的位置
            boxes_scores: 物体框的分数，为置信度和类别概率的乘积
        """
        #获得特征
        boxXY, boxWH, boxConfidence, boxClassProbs = self._getFeats(feats,anchors, classesNum, inputShape)
        boxes = self.correctBoxes(boxXY, boxWH, inputShape, imageShape)
        boxes = tf.reshape(boxes, [-1, 4])
        #获得置信度
        boxScores = boxConfidence * boxClassProbs
        boxScores = tf.reshape(boxScores, [-1, classesNum])
        return boxes, boxScores
    #解码过程
    def _getFeats(self, feats, anchors, numClasses, inputShape):
        """
        Introduction
        ------------
            根据yolo最后一层的输出确定bounding box
        Parameters
        ----------
            feats: yolo模型最后一层输出
            anchors: anchors的位置
            numClasses: 类别数量
            inputShape: 输入大小
        Returns
        -------
            boxXY, boxWH, boxConfidence, boxClassProbs
        """
        numAnchors = len(anchors)
        anchorsTensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, numAnchors, 2])
        gridSize = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, gridSize[0], gridSize[1], numAnchors, numClasses + 5])
        # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标 tile按[1, gridSize[1], 1, 1]进行平铺复制
        gridY = tf.tile(tf.reshape(tf.range(gridSize[0]), [-1, 1, 1, 1]), [1, gridSize[1], 1, 1])
        gridX = tf.tile(tf.reshape(tf.range(gridSize[1]), [1, -1, 1, 1]), [gridSize[0], 1, 1, 1])
        grid = tf.concat([gridX, gridY], axis = -1)
        grid = tf.cast(grid, tf.float32)

        # 将x,y坐标归一化，相对网格的位置
        boxXY = (tf.sigmoid(predictions[...,:2]) + grid) / tf.cast(gridSize[::-1], tf.float32)
        # 将w,h也归一化
        boxWH = tf.exp(predictions[..., 2:4]) * anchorsTensor / tf.cast(inputShape[::-1], tf.float32)
        boxConfidence = tf.sigmoid(predictions[..., 4:5])
        boxClassProbs = tf.sigmoid(predictions[..., 5:])
        return boxXY, boxWH, boxConfidence, boxClassProbs
    #获得在原图上框的位置
    def correctBoxes(self, boxXY, boxWH, inputShape, imageShape):
       """
       Introduction
       ------------
           计算物体框预测坐标在原图中的位置坐标
       Parameters
       ----------
           boxXY: 物体框左上角坐标
           boxWH: 物体框的宽高
           inputShape: 输入的大小
           imageShape: 图片的大小
       Returns
       -------
           boxes: 物体框的位置
       """
       boxYX = boxXY[..., ::-1]
       boxHW = boxWH[..., ::-1]
       #416, 416
       inputShape = tf.cast(inputShape, dtype = tf.float32)
       imageShape = tf.cast(imageShape, dtype = tf.float32)
       newShape = tf.round(imageShape * tf.reduce_min(inputShape / imageShape))

       offset = (inputShape - newShape) / 2. / inputShape
       scale = inputShape / newShape
       boxYX = (boxYX - offset) * scale
       boxHW *= scale

       boxMins = boxYX - (boxHW / 2.)
       boxMaxes = boxYX + (boxHW / 2.)
       boxes = tf.concat([

           boxMins[..., 0:1],
           boxMins[..., 1:2],
           boxMaxes[..., 0:1],
           boxMaxes[..., 1:2]
       ], axis = -1)
       boxes *= tf.concat([imageShape, imageShape], axis = -1)
       return boxes

    def eval(self, yoloOutputs, imageShape, maxBoxes = 20):
        """
        Introduction
        ------------
            根据Yolo模型的输出进行非极大值抑制，获取最后的物体检测框和物体检测类别
        Parameters
        ----------
            yoloOutpusutputs: yolo模型输出
            imageShape: 图片的大小
            maxBoxes:  最大box数量
        Returns
        -------
            boxes_: 物体框的位置
            scores_: 物体类别的概率
            classes_: 物体类别
        """
        # 每一个特征层对应三个先验框
        anchorsMask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        boxScores = []
        # inputshape是416x416
        # image_shape是实际图片的大小
        inputShape = tf.shape(yoloOutputs[0])[1 : 3] * 32
        # 对三个特征层的输出获取每个预测box坐标和box的分数，score = 置信度x类别概率
        for i in range(len(yoloOutputs)):
            _boxes, _boxesScores = self.boxesAndScores(yoloOutputs[i], self.anchors[anchorsMask[i]],
                                                       len(self.classesNames), inputShape, imageShape)
            boxes.append(_boxes)
            boxScores.append(_boxesScores)
        # 放在一行里面便于操作
        boxes = tf.concat(boxes, axis = 0)
        boxScores = tf.concat(boxScores, axis = 0)

        mask = boxScores >= self.objThreshold
        maxBoxesTensor = tf.constant(maxBoxes, dtype = tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []
        # ---------------------------------------#
        #   1、取出每一类得分大于self.obj_threshold
        #   的框和得分
        #   2、对得分进行非极大抑制
        # ---------------------------------------#
        #判断 每一个类
        for c in range(len(self.classesNames)):
            # 取出所有类为c的box
            classBoxes = tf.boolean_mask(boxes, mask[:, c])
            # 取出所有类为c的分数
            classBoxScores = tf.boolean_mask(boxScores[:, c], mask[:, c])
            nmsIndex = tf.image.non_max_suppression(classBoxes, classBoxScores, maxBoxesTensor,
                                                    iou_threshold = self.nmsThreshold)
            #获取非极大抑制的结果
            classBoxes = tf.gather(classBoxes, nmsIndex)
            classBoxScores = tf.gather(classBoxScores, nmsIndex)
            classes = tf.ones_like(classBoxScores, 'int32') * c

            boxes_.append(classBoxes)
            scores_.append(classBoxScores)
            classes_.append(classes)
        #按行拼接
        boxes_ = tf.concat(boxes_, axis = 0)
        scores_ = tf.concat(scores_, axis = 0)
        classes_ = tf.concat(classes_, axis = 0)
        return boxes_, scores_, classes_

    #---------------------------------------#
    #   predict用于预测，分三步
    #   1、建立yolo对象
    #   2、获得预测结果
    #   3、对预测结果进行处理
    #---------------------------------------#
    def predict(self, inputs, imageShape):
        """
        Introduction
        ------------
            构建预测模型
        Parameters
        ----------
            inputs: 处理之后的输入图片
            imageShape: 图像原始大小
        Returns
        -------
            boxes: 物体框坐标
            scores: 物体概率值
            classes: 物体类别
        """
        model = yolo(config.normEpsilon, config.normDecay, self.anchorsPath, self.classesPath, preTrain = False)
        # yoloInference用于获得网络的预测结果
        output = model.yoloInference(inputs, config.numAnchors // 3, config.numClasses, training = False)
        boxes, scores, classes = self.eval(output, imageShape, maxBoxes = 20)
        return boxes, scores, classes