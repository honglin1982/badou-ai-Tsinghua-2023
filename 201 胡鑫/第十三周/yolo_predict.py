import tensorflow as tf
import numpy as np
import config
import colorsys
import random
import os
from model.yolo3_model import yolo


class yolo_predictor:
    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        """
        初始化函数
        :param obj_threshold:目标检测为物体的阈值
        :param nms_threshold:nms阈值
        :param classes_file:分类标签文件
        :param anchors_file:先验框尺寸文件
        """
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold

        self.classes_path = classes_file
        self.anchors_path = anchors_file
        # 读取种类名称
        self.class_names = self._get_class()
        # 读取先验框尺寸
        self.anchors = self._get_anchors()

        # 用于画框
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]

        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)

    def _get_class(self):
        """
        读取类别名称
        :return:
        """
        # 转换为当前操作系统可识别的有效路径
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        """
        读取先验框数据
        :return:
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        """
        将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数
        :param feats: yolo输出的各个feature map
        :param anchors: anchor的位置
        :param classes_num: 类别数目
        :param input_shape: 输入大小
        :param image_shape: 图片大小
        :return:
        boxes: 物体框的位置
        boxes_scores: 物体框的分数，为置信度x类别概率
        """
        # 获得特征
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num,
                                                                          input_shape)
        # 寻找在原图上的位置
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        # 获取box_confidence * box_class_probs
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        """
        计算物体框预测坐标在原图中的坐标位置
        :param box_xy: 物体框左上角坐标
        :param box_wh: 物体框的宽高
        :param input_shape: 输入的大小
        :param image_shape: 图片的大小
        :return:
        boxes: 物体框的位置
        """
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        # 416x416
        input_shape = tf.cast(input_shape, dtype=tf.float32)
        # 实际图片大小
        image_shape = tf.cast(image_shape, dtype=tf.float32)
        # 以最小比率缩放，保存图象完整性
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        # 最终的 offset 值表示了在将图像放置在 input_shape 画布上时，需要在图像周围添加的边框的比例。
        # 这个偏移量可以用于在图像上应用相应的平移操作，以使图像居中放置在 input_shape 中，以便进行后续的处理或模型输入。
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        # 映射到new_shape，变换成中心坐标
        box_yx = (box_yx - offset) * scale
        box_hw *= scale
        # 变换到左上右下两个坐标
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)
        # 映射回原始的image_shape
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    def _get_feats(self, feats, anchors, num_classes, input_shape):
        """
        根据yolo最后一层的输出确定bounding box，解码过程
        :param feats: yolo模型最后一层输出
        :param anchors: anchors的位置
        :param num_classes: 类别数量
        :param input_shape: 输入大小
        :return: box的各个特征
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        # 结合ppt中的图的样子的形式
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        # 例如第一个特征层构建13,13,1,2的矩阵，对应每个格子加上对应的坐标(每个网格的相对坐标)
        # 两个都是13,13,1,1
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)

        # 对最后一个维度进行操作，最后一个维度所包含信息顺序为坐标、尺寸、置信度、各类别概率
        # 得到预测出的框的坐标在网格图中的相对坐标
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # wh一样,之所以会交换shape，是因为储存信息的时候是xywh这样子的顺序，需要将hw交换成wh
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        # 后面都使用sigmoid归一化
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs

    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        """
        根据yolo模型的输出进行非极大值抑制，获取最后的物体检测框，和物体检测类别
        :param yolo_outputs: yolo模型输出
        :param image_shape: 图片的大小
        :param max_boxes: 最大box数量
        :return:
        boxes_: 物体框的位置
        scores_: 物体类别的概率
        classes_: 物体的类别
        """
        # 每一个特征层对应三个先验框
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        # input_shape是416x416
        # image_shape是实际图片大小
        input_shape = tf.shape(yolo_outputs[0])[1:3] * 32  # 13*32=416
        # 对三个特征层的输出获取每个预测box坐标和box的分数：score = 置信度x类别概率
        # ------------------------------------ #
        #       对三个特征层解码
        #       获取分数和框的位置
        # ------------------------------------ #
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]],
                                                        len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        # 放在一行里面便于操作
        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        mask = box_scores >= self.obj_threshold
        # 满足条件后从高到低选，最高保留max_boxes_tensor个框
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []
        # ---------------------------------------#
        #   1、取出每一类得分大于self.obj_threshold
        #   的框和得分
        #   2、对得分进行非极大抑制
        # ---------------------------------------#
        # 对每一个类进行判断
        for c in range(len(self.class_names)):
            # 取出所有类为c的box
            # mask[:, c]表示类别为c的物体框的掩码。
            # 匹配上了将其取出，class_boxes表示所有类别为c的物体框的坐标信息
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            # 取出所有 类别为c的分数
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            # 非极大值抑制
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                     iou_threshold=self.nms_threshold)
            # 获取非极大值抑制的结果
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            # 这行代码用于创建一个形状与 class_box_scores 张量相同的新张量 classes，其中所有的元素都被设置为整数 c。
            # 整数 c 表示一个类别的标识符
            classes = tf.ones_like(class_box_scores, 'int32') * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)
        return boxes_, scores_, classes_

    def predict(self, inputs, image_shape):
        """
        构建预测模型
        :param inputs: 处理之后的输入图片
        :param image_shape: 图象原始大小
        :return:
        boxes: 物体框坐标
        scores: 物体概率值
        classes: 物体类别
        """
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path,
                     self.classes_path, pre_train=False)
        output = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training=False)
        boxes, scores, classes = self.eval(output, image_shape, max_boxes=20)
        return boxes, scores, classes

