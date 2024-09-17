import cv2
import keras
import numpy as np
import colorsys
import pickle
import os
import nets.frcnn as frcnn
from nets.frcnn_training import get_new_img_size
from keras import backend as K
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image,ImageFont, ImageDraw
from utils.utils import BBoxUtility
from utils.anchors import get_anchors
from utils.config import Config
import copy
import math

#本文件是用于检测图片的，如果想要检测视频、摄像头或者是文件夹下的所有图片，可以参考predict.py

class FRCNN(object):
    _defaults = {
        "model_path": 'model_data/voc_weights.h5',
        "classes_path": 'model_data/voc_classes.txt',
        "confidence": 0.7,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化faster RCNN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.sess = K.get_session()
        self.config = Config()                                      # 获得配置文件
        self.generate()
        self.bbox_util = BBoxUtility()                              # 生成工具箱
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 计算总的种类
        self.num_classes = len(self.class_names)+1

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        self.model_rpn,self.model_classifier = frcnn.get_predict_model(self.config,self.num_classes)    # 获得模型
        self.model_rpn.load_weights(self.model_path,by_name=True)                                       # 载入权重
        self.model_classifier.load_weights(self.model_path,by_name=True,skip_mismatch=True)             # 加载权重
                
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
    
    def get_img_output_length(self, width, height):                                                     # 获得图片的长宽
        def get_output_length(input_length):
            # input_length += 6
            filter_sizes = [7, 3, 1, 1]
            padding = [3,1,0,0]
            stride = 2
            for i in range(4):
                # input_length = (input_length - filter_size + stride) // stride
                input_length = (input_length+2*padding[i]-filter_sizes[i]) // stride + 1
            return input_length
        return get_output_length(width), get_output_length(height)                                      # 获得图片的长宽
    
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        old_width = image_shape[1]
        old_height = image_shape[0]
        old_image = copy.deepcopy(image)
        width,height = get_new_img_size(old_width,old_height)


        image = image.resize([width,height])
        photo = np.array(image,dtype = np.float64)

        # 图片预处理，归一化
        photo = preprocess_input(np.expand_dims(photo,0))                                               # 图片预处理，归一化
        preds = self.model_rpn.predict(photo)                                                           # 将预测结果进行解码
        # 将预测结果进行解码
        anchors = get_anchors(self.get_img_output_length(width,height),width,height)                    # 获得anchor，用于解码

        rpn_results = self.bbox_util.detection_out(preds,anchors,1,confidence_threshold=0)              # 将预测结果进行解码，获得预测框
        R = rpn_results[0][:, 2:]                                                                       # 获得预测框的坐标和得分，R的shape为[num,(x1,y1,x2,y2,score)]
        
        R[:,0] = np.array(np.round(R[:, 0]*width/self.config.rpn_stride),dtype=np.int32)                # 将预测框的坐标转换成原图的坐标，R[:0]表示x1
        R[:,1] = np.array(np.round(R[:, 1]*height/self.config.rpn_stride),dtype=np.int32)               # 将预测框的坐标转换成原图的坐标，R[:1]表示y1
        R[:,2] = np.array(np.round(R[:, 2]*width/self.config.rpn_stride),dtype=np.int32)                # 将预测框的坐标转换成原图的坐标，R[:2]表示x2
        R[:,3] = np.array(np.round(R[:, 3]*height/self.config.rpn_stride),dtype=np.int32)               # 将预测框的坐标转换成原图的坐标，R[:3]表示y2
        
        R[:, 2] -= R[:, 0]                                                                              # R[:2]表示x2-x1，即宽度
        R[:, 3] -= R[:, 1]                                                                              # 将预测框的坐标转换成原图的坐标，R[:3]表示y2-y1
        base_layer = preds[2]                                                                           # 获得预测结果中的第三个结果，即feature map
        
        delete_line = []                                                                                # 删除一些不合规的框
        for i,r in enumerate(R):                                                                        # 遍历所有的预测框
            if r[2] < 1 or r[3] < 1:                                                                    # 如果预测框的宽度或者高度小于1
                delete_line.append(i)                                                                   # 将这个预测框的索引添加到delete_line中
        R = np.delete(R,delete_line,axis=0)                                                             # 删除这些不合规的预测框
        
        bboxes = []
        probs = []
        labels = []
        for jk in range(R.shape[0]//self.config.num_rois + 1):                                          # 遍历所有的预测框，每次遍历128个
            ROIs = np.expand_dims(R[self.config.num_rois*jk:self.config.num_rois*(jk+1), :], axis=0)    # 获得一定数量的预测框
            
            if ROIs.shape[1] == 0:                                                                      # 如果没有预测框，则退出
                break

            if jk == R.shape[0]//self.config.num_rois:                                                  # 如果是最后一次，而且不足128个,则补齐128个
                #pad R
                curr_shape = ROIs.shape                                                                 # 获得当前的shape
                target_shape = (curr_shape[0],self.config.num_rois,curr_shape[2])                       # 获得目标shape
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)                                 # 创建一个全0的数组
                ROIs_padded[:, :curr_shape[1], :] = ROIs                                                # 将ROIs的值复制到ROIs_padded中
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]                                       # 将最后的几个预测框复制到最后，补齐128个
                ROIs = ROIs_padded                                                                      # 将ROIs_padded赋值给ROIs
            
            [P_cls, P_regr] = self.model_classifier.predict([base_layer,ROIs])                         # 获得预测框的类别和回归系数，类别是21个，回归系数是4个，即x,y,w,h

            for ii in range(P_cls.shape[1]):                                                                            # 遍历所有的预测框
                if np.max(P_cls[0, ii, :]) < self.confidence or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):     # 如果这个预测框的最大得分小于confidence，或者这个预测框的类别为背景，则跳过这个预测框
                    continue

                label = np.argmax(P_cls[0, ii, :])                                                                    # 获得这个预测框的类别

                (x, y, w, h) = ROIs[0, ii, :]                                                                     # 获得这个预测框的坐标，注意，这个坐标是在feature map上的坐标，而不是在原图上的坐标

                cls_num = np.argmax(P_cls[0, ii, :])                                                              # 获得这个预测框的类别

                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]                                         # 获得这个预测框的回归系数
                tx /= self.config.classifier_regr_std[0]                                                          # 进行反标准化，这里的标准化是在训练的时候进行的
                ty /= self.config.classifier_regr_std[1]
                tw /= self.config.classifier_regr_std[2]
                th /= self.config.classifier_regr_std[3]

                cx = x + w/2.                                                                                      # 获得预测框的中心坐标,
                cy = y + h/2.                                                                                       # 获得预测框的中心坐标
                cx1 = tx * w + cx                                                                                   # 利用回归系数获得预测框的中心坐标
                cy1 = ty * h + cy                                                                                   # 利用回归系数获得预测框的中心坐标
                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h

                x1 = cx1 - w1/2.
                y1 = cy1 - h1/2.

                x2 = cx1 + w1/2
                y2 = cy1 + h1/2

                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))

                bboxes.append([x1,y1,x2,y2])                                                            # 将预测框的坐标和得分添加到bboxes中
                probs.append(np.max(P_cls[0, ii, :]))                                                   # 将预测框的得分添加到probs中
                labels.append(label)                                                                    # 将预测框的类别添加到labels中

        if len(bboxes)==0:                                                                              # 如果没有一个预测框，则返回原图
            return old_image
        
        # 筛选出其中得分高于confidence的框
        labels = np.array(labels)                                                                   # 将labels转换成numpy数组
        probs = np.array(probs)                                                                    # 将probs转换成numpy数组
        boxes = np.array(bboxes,dtype=np.float32)                                               # 将bboxes转换成numpy数组
        boxes[:,0] = boxes[:,0]*self.config.rpn_stride/width                                    # 将预测框的坐标转换成原图的坐标
        boxes[:,1] = boxes[:,1]*self.config.rpn_stride/height                                   # 将预测框的坐标转换成原图的坐标
        boxes[:,2] = boxes[:,2]*self.config.rpn_stride/width
        boxes[:,3] = boxes[:,3]*self.config.rpn_stride/height
        results = np.array(self.bbox_util.nms_for_out(np.array(labels),np.array(probs),np.array(boxes),self.num_classes-1,0.4))  # 利用非极大值抑制去除重复的预测框
        
        top_label_indices = results[:,0]                                                        # 获得最终的预测框的类别
        top_conf = results[:,1]                                                                 # 获得最终的预测框的得分
        boxes = results[:,2:]                                                                   # 获得最终的预测框的坐标
        boxes[:,0] = boxes[:,0]*old_width                                                     # 将预测框的坐标转换成原图的坐标
        boxes[:,1] = boxes[:,1]*old_height
        boxes[:,2] = boxes[:,2]*old_width
        boxes[:,3] = boxes[:,3]*old_height

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        
        thickness = (np.shape(old_image)[0] + np.shape(old_image)[1]) // width
        image = old_image                                                                                # 获得原图
        for i, c in enumerate(top_label_indices):                                       # 遍历所有的预测框
            predicted_class = self.class_names[int(c)]                                # 获得预测框的类别
            score = top_conf[i]

            left, top, right, bottom = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)  # 得分和类别
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def close_session(self):
        self.sess.close()
