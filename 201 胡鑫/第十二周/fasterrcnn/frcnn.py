import cv2
import keras
import numpy as np
import colorsys
import pickle
import os
import nets.frcnn as frcnn
from nets.frcnn_training import get_new_img_size
from keras import backend as k
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image, ImageFont, ImageDraw
from utils.utils import BBoxUtility
from utils.anchors import get_anchors
from utils.config import Config
import copy
import math


class FRCNN(object):
    _defaults = {
        "model_path": "model_data/voc_weights.h5",
        "classes_path": "model_data/voc_classes.txt",
        "confidence": .7
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化faster RCNN
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.sess = k.get_session()
        self.config = Config()
        self.generate()
        self.bbox_util = BBoxUtility()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(".h5"), "Keras model or weights must be a .h5 file"

        # 计算总的种类
        self.num_classes = len(self.class_names) + 1

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入
        # 否则先构造模型再载入
        self.model_rpn, self.model_classifier = frcnn.get_predict_model(self.config, self.num_classes)
        self.model_rpn.load_weights(model_path, by_name=True)
        self.model_classifier.load_weights(model_path, by_name=True, skip_mismatch=True)

        print(f'{model_path} model, anchors, and classes loaded')

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), self.colors))

    def get_img_output_length(self, width, height):
        """计算经过base网络后的宽高"""
        def get_output_length(input_length):
            filters_sizes = [7, 3, 1, 1]
            padding = [3, 1, 0, 0]
            stride = 2
            for i in range(4):
                # input_length = (input_length - filter_size + stride) // stride
                input_length = (input_length + 2*padding[i] - filters_sizes[i]) // stride + 1
            return input_length
        return get_output_length(width), get_output_length(height)

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        old_width = image_shape[1]
        old_height = image_shape[0]
        old_image = copy.deepcopy(image)
        width, height = get_new_img_size(old_width, old_height)

        image = image.resize([width, height])
        photo = np.array(image, dtype=np.float64)

        # 图片预处理，归一化(先三维变四维再归一化)
        photo = preprocess_input(np.expand_dims(photo, 0))
        preds = self.model_rpn.predict(photo)
        # 将预测结果进行解码
        # 暴力枚举原始提案框
        anchors = get_anchors(self.get_img_output_length(width, height), width, height)
        # 二分类，这个解码的操作是将有概率有目标的提案框筛选出来
        rpn_results = self.bbox_util.detection_out(preds, anchors, 1, confidence_threshold=0)
        # 取出所有与坐标有关的信息
        R = rpn_results[0][:, 2:]
        # 将坐标信息映射回特征图谱
        R[:, 0] = np.array(np.round(R[:, 0]*width/self.config.rpn_stride), dtype=np.int32)
        R[:, 1] = np.array(np.round(R[:, 1]*height/self.config.rpn_stride), dtype=np.int32)
        R[:, 2] = np.array(np.round(R[:, 2]*width/self.config.rpn_stride), dtype=np.int32)
        R[:, 3] = np.array(np.round(R[:, 3]*height/self.config.rpn_stride), dtype=np.int32)
        # 将xmaxymax转换成宽度
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        base_layer = preds[2]

        delete_line = []
        for i, r in enumerate(R):
            if r[2] < 1 or r[3] < 1:
                delete_line.append(i)
        R = np.delete(R, delete_line, axis=0)

        bboxes = []
        probs = []
        labels = []
        # 分批次进行
        for jk in range(R.shape[0] // self.config.num_rois + 1):
            rois = np.expand_dims(R[self.config.num_rois*jk:self.config.num_rois*(jk+1), :], axis=0)
            if rois.shape[1] == 0:
                break
            if jk == R.shape[0] // self.config.num_rois:
                # 当不够一个批次的时候，填充成一个批次的数量
                # pad R
                curr_shape = rois.shape
                target_shape = (curr_shape[0], self.config.num_rois, curr_shape[2])
                rois_padded = np.zeros(target_shape).astype(rois.dtype)
                rois_padded[:, :curr_shape[1], :] = rois
                rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]
                rois = rois_padded
            # 分批次传递给分类网络
            [p_cls, p_regr] = self.model_classifier.predict([base_layer, rois])
            # p_cls.shape = (1, num_rois, nb_classes)
            for ii in range(p_cls.shape[1]):
                if np.max(p_cls[0, ii, :]) < self.confidence or \
                        np.argmax(p_cls[0, ii, :]) == (p_cls.shape[2] - 1):  # 最大概率的类别为背景
                    continue
                # 当前第ii个预测框最大概率的类别的索引
                label = np.argmax(p_cls[0, ii, :])
                # rois.shape = (1, num_rois, 4)
                x, y, w, h = rois[0, ii, :]
                # 当前第ii个预测框最大概率的类别的索引（模型认为的当前预测框的类别）
                cls_num = np.argmax(p_cls[0, ii, :])
                # 按照对应类别获取对应的四个坐标信息（相对于rois的坐标有偏移）
                tx, ty, tw, th = p_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                # 这个过程的目的是将位置回归信息还原到原始的尺度，以便将其应用
                # 于当前提案框，以微调提案框的位置和大小
                tx /= self.config.classifier_regr_std[0]
                ty /= self.config.classifier_regr_std[1]
                tw /= self.config.classifier_regr_std[2]
                th /= self.config.classifier_regr_std[3]
                # 有目标的原始提案框（第一次筛选后的rois）的中心坐标
                cx = x + w / 2.
                cy = y + h / 2.
                # 经过分类网络后的各个框的中心坐标
                cx1 = tx * w + cx
                cy1 = ty * h + cy
                # 微调后的宽高（加指数增加学习效率（网络容易学习到差别更大的结果））
                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h
                # 微调后的四个坐标值
                x1 = cx1 - w1 / 2.
                y1 = cy1 - h1 / 2.
                x2 = cx1 + w1 / 2
                y2 = cy1 + h1 / 2

                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))

                bboxes.append([x1, y1, x2, y2])  # 四个坐标值
                probs.append(np.max(p_cls[0, ii, :]))  # 最大概率的值
                labels.append(label)  # 最大概率的值的索引，对应有类别

        if len(bboxes) == 0:
            return old_image

        # 筛选出其中得分高于confidence的框
        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(bboxes, dtype=np.float32)
        # 将坐标信息映射回原图
        boxes[:, 0] = boxes[:, 0] * self.config.rpn_stride / width
        boxes[:, 1] = boxes[:, 1] * self.config.rpn_stride / height
        boxes[:, 2] = boxes[:, 2] * self.config.rpn_stride / width
        boxes[:, 3] = boxes[:, 3] * self.config.rpn_stride / height

        results = np.array(self.bbox_util.nms_for_out(np.array(labels), np.array(probs),
                                                      np.array(boxes), self.num_classes-1, .4))
        top_label_indices = results[:, 0]
        top_conf = results[:, 1]
        boxes = results[:, 2:]
        # 映射回读取的图片
        boxes[:, 0] = boxes[:, 0] * old_width
        boxes[:, 1] = boxes[:, 1] * old_height
        boxes[:, 2] = boxes[:, 2] * old_width
        boxes[:, 3] = boxes[:, 3] * old_height

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + .5).astype('int32'))

        thickness = (np.shape(old_image)[0] + np.shape(old_image)[1]) // width
        image = old_image

        for i, c in enumerate(top_label_indices):
            # 获取当前检测目标的类别名称 predicted_class 和置信度分数 score
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]
            # 对坐标进行微调，增加或减少一些像素，以便更好地框住目标。
            left, top, right, bottom = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            # 确保坐标在图像边界内。
            top = max(0, np.floor(top + .5).astype('int32'))
            left = max(0, np.floor(left + .5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + .5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + .5).astype('int32'))

            # 画框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j],
                               outline=self.colors[int(c)])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
                           fill=self.colors[int(c)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def close_session(self):
        self.sess.close()






