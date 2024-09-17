import copy
import math
import numpy as np
import os
import colorsys
import nets.frcnn as frcnn
from nets.frcnnTraining import getNewImgSize
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from PIL import  ImageFont, ImageDraw
from utils.config import Config
from utils.anchors import getAnchors
from utils.utils import  BBoxUtility

class FRCNN(object):

    _defaults = {
        "model_path" : 'model_data/voc_weights.h5',
        "classes_path" : 'model_data/voc_classes.txt',
        "confidence": 0.7,
    }
    @classmethod
    def getDefaults(cls, n):
        return cls._defaults[n] if n in cls._defaults else "Unrecognized attribute name '" + n + "'"
    #初始化 faster RCNN
    def __init__(self):

        self.__dict__.update(self._defaults)
        self.classNames = self._getClass()
        self.sess = K.get_session()
        self.config = Config()
        self.generate()
        self.bboxUtil = BBoxUtility()
    #获取的所有分类
    def _getClass(self):

        classesPath = os.path.expanduser(self.classes_path)
        with open(classesPath) as f : classesName = f.readlines()
        classesName = [c.strip() for c in classesName]
        return classesName
    #获取所有分类
    def generate(self):

        modelPath = os.path.expanduser(self.model_path)
        assert modelPath.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        #计算总的种类
        self.numClasses = len(self.classNames) + 1
        self.modelRpn, self.modelClassifier = frcnn.getPredictModel(self.config, self.numClasses)
        self.modelRpn.load_weights(self.model_path, by_name=True)
        self.modelClassifier.load_weights(self.model_path, by_name=True, skip_mismatch=True)
        print('{} model, anchors, and classes loaded.'.format(modelPath))

        # 画框设置不同的颜色
        hsvTuples = [(x / len(self.classNames), 1., 1.) for x in range(len(self.classNames))]
        self.colors = list(map(lambda x : colorsys.hsv_to_rgb(*x), hsvTuples))
        self.colors = list(map(lambda x : (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def getImgOutputLength(self, width, height):
        def getOutputLength(inputLength):
            # input_length += 6
            filterSizes = [7, 3, 1, 1]
            padding = [3, 1, 0, 0]
            stride = 2
            for i in range(4):
                # input_length = (input_length - filter_size + stride) // stride
                inputLength = (inputLength + 2 * padding[i] - filterSizes[i]) // stride + 1
            return inputLength
        return getOutputLength(width), getOutputLength(height)

    # 检测图片
    def detectImage(self, image):

        imageShape = np.array(np.shape(image)[0:2])
        oldWidth = imageShape[1]
        oldHeight = imageShape[0]
        oldImage = copy.deepcopy(image)
        width, height = getNewImgSize(oldWidth, oldHeight)

        image = image.resize([width, height])
        photo = np.array(image, dtype=np.float64)

        #图片预处理 归一化
        photo = preprocess_input(np.expand_dims(photo, 0))
        preds = self.modelRpn.predict(photo)
        #将预测结果进行解码
        anchors = getAnchors(self.getImgOutputLength(width, height),
                             width, height)
        rpnResults = self.bboxUtil.detectionOut(preds, anchors, 1, confidenceThreshold=0)
        R = rpnResults[0][:, 2:]

        R[:, 0] = np.array(np.round(R[:, 0] * width / self.config.rpnStride), dtype=np.int32)
        R[:, 1] = np.array(np.round(R[:, 1] * width / self.config.rpnStride), dtype=np.int32)
        R[:, 2] = np.array(np.round(R[:, 2] * width / self.config.rpnStride), dtype=np.int32)
        R[:, 3] = np.array(np.round(R[:, 3] * width / self.config.rpnStride), dtype=np.int32)

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        baseLayer = preds[2]

        deleteLine = []
        for i, r in enumerate(R):
            if r[2] < 1 or r[3] < 1:
                deleteLine.append(i)
        R = np.delete(R, deleteLine, axis=0)

        bboxes = []
        probs = []
        labels = []
        for jk in range(R.shape[0] //self.config.numRois + 1):

            ROIS = np.expand_dims(R[self.config.numRois * jk : self.config.numRois * (jk + 1), :], axis=0)
            if ROIS.shape[1] == 0: break
            if jk == R.shape[0] // self.config.numRois :
                currShape = ROIS.shape
                targetShape = (currShape[0], self.config.numRois, currShape[2])
                ROISpadded = np.zeros(targetShape).astype(ROIS.dtype)
                ROISpadded[:, :currShape[1], :] = ROIS
                ROISpadded[0, currShape[1]:, :] = ROIS[0, 0, :]
                ROIS = ROISpadded
            [Pcls, Pregr] = self.modelClassifier.predict([baseLayer, ROIS])

            for ii in range(Pcls.shape[1]):

                if np.max(Pcls[0, ii, :]) < self.confidence or \
                        np.argmax(Pcls[0, ii, :]) == (Pcls.shape[2] - 1): continue

                label = np.argmax(Pcls[0, ii, :])
                (x, y, w, h) = ROIS[0, ii, :]
                clsNum = np.argmax(Pcls[0, ii, :])
                (tx, ty, tw, th) = Pregr[0, ii, 4 * clsNum : 4 * (clsNum + 1)]
                tx /= self.config.classifierRegrStd[0]
                ty /= self.config.classifierRegrStd[1]
                tw /= self.config.classifierRegrStd[2]
                th /= self.config.classifierRegrStd[3]

                cx, cy = x + w/2., y + h/2.
                cx1, cy1 = tx * w + cx, ty * h + cy
                w1, h1 = math.exp(tw) * w, math.exp(th) * h

                x1, y1 = cx1 - w1/2., cy1 - h1/2.
                x2, y2 = cx1 + w1/2., cy1 + h1/2.
                x1, y1 = int(round(x1)), int(round(y1))
                x2, y2 = int(round(x2)), int(round(y2))

                bboxes.append([x1, y1, x2, y2])
                probs.append(np.max(Pcls[0, ii, :]))
                labels.append(label)
        if len(bboxes) == 0: return oldImage

        # 筛选出其中得分高于confidence的框
        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(bboxes, dtype=np.float32)
        boxes[:,0] = boxes[:,0] * self.config.rpnStride / width
        boxes[:,1] = boxes[:,1] * self.config.rpnStride / height
        boxes[:,2] = boxes[:,2] * self.config.rpnStride / width
        boxes[:,3] = boxes[:,3] * self.config.rpnStride / height
        results = np.array(self.bboxUtil.nmsForOut(np.array(labels), np.array(probs),
                                                   np.array(boxes), self.numClasses - 1, 0.4))
        topLableIndices = results[:,0]
        topConf = results[:,1]
        boxes = results[:,2:]
        boxes[:,0] = boxes[:,0] * oldWidth
        boxes[:,1] = boxes[:,1] * oldHeight
        boxes[:,2] = boxes[:,2] * oldWidth
        boxes[:,3] = boxes[:,3] * oldHeight
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = (np.shape(oldImage)[0] + np.shape(oldImage)[1]) // width
        image = oldImage
        for i, c in enumerate(topLableIndices):

            predictedClass = self.classNames[int(c)]
            score = topConf[i]
            left, top, right, bottom = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            #画框
            label = '{} {:.2f}'.format(predictedClass, score)
            draw = ImageDraw.Draw(image)
            labelSize = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            if top - labelSize[1] >= 0 :
                textOrigin = np.array([left, top - labelSize[1]])
            else:
                textOrigin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[int(c)])

            draw.rectangle([tuple(textOrigin), tuple(textOrigin + labelSize)], fill=self.colors[int(c)])
            draw.text(textOrigin, str(label, 'utf-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def closeSession(self): self.sess.close()



