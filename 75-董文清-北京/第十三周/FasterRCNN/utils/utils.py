import select
import tensorflow as tf
import numpy as np
from PIL import Image
import keras
import math

class BBoxUtility(object):

    def __init__(self, priors=None, overlapThrehold=0.7, ignoreThreshold=0.3,
                 nmsThresh=0.7, topK=300):
        self.priors = priors
        self.numPriors = 0 if priors is None else len(priors)
        self.overlapThreshold = overlapThrehold
        self.ignoreThreshold = ignoreThreshold
        self._nmsThresh = nmsThresh
        self._topK = topK
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores, self.topK,
                                                iou_threshold=self.nmsThresh)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':0}))

    #nmsThresh 和 topK 的 getter setter 方法
    @property
    def nmsThresh(self): return self._nmsThresh
    @nmsThresh.setter
    def nmsThresh(self, value):
        self._nmsThresh = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._topK, iou_threshold=self._nmsThresh)
    @property
    def topK(self): return self._topK
    @topK.setter
    def topK(self, value):
        self._topK = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores, self._topK,
                                                iou_threshold=self._nmsThresh)

    def iou(self, box):

        #计算出每个真实框的iou
        #判断真实框和先验框的重合情况
        interUpleft = np.maximum(self.priors[:, :2], box[:2])
        interBotright = np.minimum(self.priors[:, 2:4], box[2:])

        interWh = interBotright - interUpleft
        interWh = np.maximum(interWh, 0)
        inter = interWh[:, 0] * interWh[:, 1]
        #真实框面积
        areaTrue = (box[2] - box[0]) * (box[3] - box[1])
        #先验框面积
        areaGt = (self.priors[:, 2] - self.priors[:, 0]) * (self.priors[:, 3] - self.priors[:, 1])
        union = areaTrue + areaGt - inter
        iou = inter / union
        return iou

    def encodeBox(self, box, returnIou=True):

        iou = self.iou(box)
        encodeBox = np.zeros((self.priors, 4 + returnIou))

        #找到与真实框重合度较高的先验框
        assignMask = iou > self.overlapThreshold
        if not assignMask.any(): assignMask[iou.argmax()] = True
        if returnIou: encodeBox[:, -1][assignMask] = iou[assignMask]

        #找对应的先验框
        assignedPriors = self.priors[assignMask]
        # 逆向编码，将真实框转化为FasterRCNN预测结果的格式
        # 先计算真实框的中心与长宽
        boxCenter = 0.5 * (box[:2] + box[2:])
        boxWh = box[2:] - box[:2]
        assignedPriorsCenter = 0.5 * (assignedPriors[:, :2] + assignedPriors[:, 2:4])
        assignedPriorsWh = (assignedPriors[:, 2 : 4] - assignedPriors[:, :2])

        # 逆向求取FasterRCNN应该有的预测结果
        encodeBox[:, :2][assignMask] = boxCenter - assignedPriorsCenter
        encodeBox[:, :2][assignMask] /= assignedPriorsWh
        encodeBox[:, :2][assignMask] *= 4
        encodeBox[:, 2:4][assignMask] = np.log(boxWh / assignedPriorsWh)
        encodeBox[:, 2:4][assignMask] *= 4
        return encodeBox.ravel()

    def ignoreBox(self, box):

        iou = self.iou(box)
        ignoredBox = np.zeros((self.numPriors, 1))
        # 找到与真实框重合度较高的先验框
        assignMask = (iou > self.ignoreThreshold) & (iou < self.overlapThreshold)
        if not assignMask.any(): assignMask[iou.argmax()] = True
        ignoredBox[:, 0][assignMask] = iou[assignMask]
        return ignoredBox.ravel()

    def assignBoxes(self, boxes, anchors):

        self.numPriors = len(anchors)
        self.priors = anchors
        assignment = np.zeros((self.numPriors, 4 + 1))
        assignment[:, 4] = 0.0
        if len(boxes) == 0: return assignment
        #对每个真实框进行iou计算
        ingoredBoxes = np.apply_along_axis(self.ignoreBox, 1, boxes[:, :4])
        # 取重合程度最大的先验框，并且获取这个先验框的index
        ingoredBoxes = ingoredBoxes.reshape(-1, self.numPriors, 1)
        ignoreIou = ingoredBoxes[:, :, 0].max(axis=0)
        ignoreIouMask = ignoreIou > 0
        assignment[:, 4][ignoreIouMask] = -1

        # (n, num_priors, 5)
        encodedBoxes = np.apply_along_axis(self.encodeBox, 1, boxes[:, :4])
        # 每一个真实框的编码后的值，和iou
        # (n, num_priors)
        encodedBoxes = encodedBoxes.reshape(-1, self.numPriors, 5)

        # 取重合程度最大的先验框，并且获取这个先验框的index
        # (num_priors)
        bestIou = encodedBoxes[:, :, -1].max(axis=0)
        bestIouIdx = encodedBoxes[:, :, -1].argmax(axis=0)
        bestIouMask = bestIou > 0
        bestIouIdx = bestIouIdx[bestIouMask]

        assignNum = len(bestIouIdx)
        # 保留重合程度最大的先验框的应该有的预测结果
        # 哪些先验框存在真实框
        encodedBoxes = encodedBoxes[:, bestIouMask, :]

        assignment[:, :4][bestIouMask] = encodedBoxes[bestIouIdx, np.arange(assignNum), : 4]
        # 4代表为背景的概率，为0
        assignment[:, 4][bestIouMask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment

    def decodeBoxes(self, mboxLoc, mboxPriorbox):

        #获取先验框的宽高
        priorWidth = mboxPriorbox[:, 2] - mboxPriorbox[:, 0]
        priorHeight = mboxPriorbox[:, 3] - mboxPriorbox[:, 1]

        #获取先验框中心点
        priorCenterX = 0.5 * (mboxPriorbox[:, 2] + mboxPriorbox[:, 0])
        priorCenterY = 0.5 * (mboxPriorbox[:, 3] + mboxPriorbox[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decodeBBoxCenterX = mboxLoc[:, 0] * priorWidth / 4
        decodeBBoxCenterX += priorCenterX
        decodeBBoxCenterY = mboxLoc[:, 1] * priorHeight / 4
        decodeBBoxCenterY += priorCenterY
        # 真实框的宽与高的求取
        decodeBBoxWidth = np.exp(mboxLoc[:, 2] / 4)
        decodeBBoxWidth *= priorWidth
        decodeBBoxHeight = np.exp(mboxLoc[:, 3] / 4)
        decodeBBoxHeight *= priorHeight

        #获取真实框的左上角与右下角
        decodeBBoxXmin = decodeBBoxCenterX - 0.5 * decodeBBoxWidth
        decodeBBoxYmin = decodeBBoxCenterY - 0.5 * decodeBBoxHeight
        decodeBBoxXmax = decodeBBoxCenterX + 0.5 * decodeBBoxWidth
        decodeBBoxYmax = decodeBBoxCenterY + 0.5 * decodeBBoxHeight
        # 真实框的左上角与右下角进行堆叠
        decodeBBox = np.concatenate((decodeBBoxXmin[:, None], decodeBBoxYmin[:, None],
                                     decodeBBoxXmax[:, None], decodeBBoxYmax[:, None]), axis=-1)
        # 防止超出0与1
        decodeBBox = np.minimum(np.maximum(decodeBBox, 0.0), 1.0)
        return decodeBBox

    def detectionOut(self, predictions, mboxPriorbox, numClasses, keepTopK=300,
                     confidenceThreshold=0.5):
        # 网络预测的结果
        # 置信度
        mboxConf = predictions[0]
        mboxLoc = predictions[1]
        # 先验框
        mboxPriorbox = mboxPriorbox
        results = []
        #处理图片
        for i in range(len(mboxLoc)):
            results.append([])
            decodeBBox = self.decodeBoxes(mboxLoc[i], mboxPriorbox)
            for c in range(numClasses):
                cConfs = mboxConf[i, :, c]
                cConfsM = cConfs > confidenceThreshold
                if len(cConfs[cConfsM]) > 0:
                     # 取出得分高于confidence_threshold的框
                     boxesToProcess = decodeBBox[cConfsM]
                     confsToProcess = cConfs[cConfsM]
                     # 进行iou的非极大抑制
                     feedDict = {self.boxes: boxesToProcess,
                                 self.scores: confsToProcess}
                     idx = self.sess.run(self.nms, feed_dict=feedDict)
                     # 取出在非极大抑制中效果较好的内容
                     goodBoxes = boxesToProcess[idx]
                     confs = confsToProcess[idx][:, None]
                     # 将label、置信度、框的位置进行堆叠
                     labels = c * np.ones((len(idx), 1))
                     cPred = np.concatenate((labels, confs, goodBoxes), axis=1)
                     results[-1].extend(cPred) #添加到result中

                if len(results[-1]) > 0 :

                    results[-1] = np.array(results[-1])
                    argsort = np.argsort(results[-1][:, 1])[::-1]
                    results[-1] = results[-1][argsort]
                    # 选出置信度最大的keep_top_k个
                    results[-1] = results[-1][:keepTopK]
            # 获得，在所有预测结果里面，置信度比较高的框
            # 还有，利用先验框和Retinanet的预测结果，处理获得了真实框（预测框）的位置
            return results

    def nmsForOut(self, allLabels, allConfs, allBBoxes, numClasses, nms):

        results = []
        nmsOut = tf.image.non_max_suppression(self.boxes, self.scores,
                                              self._topK, iou_threshold=nms)
        for c in range(numClasses):
            cPred = []
            mask = allLabels == c
            if len(allConfs[mask]) > 0:
                # 取出得分高于confidence_threshold的框
                boxesToProcess = allBBoxes[mask]
                confsToProcess = allConfs[mask]
                # 进行iou的非极大抑制
                feedDict = {self.boxes: boxesToProcess,
                            self.scores:confsToProcess}
                idx = self.sess.run(nmsOut, feed_dict=feedDict)
                goodBoxes = boxesToProcess[idx]
                confs = confsToProcess[idx][:, None]
                # 将label、置信度、框的位置进行堆叠
                labels = c * np.ones((len(idx), 1))
                cPred = np.concatenate((labels, confs, goodBoxes), axis=1)
            results.extend(cPred)
        return results