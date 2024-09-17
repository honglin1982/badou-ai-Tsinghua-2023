import tensorflow as tf
import numpy as np
import os


class yolo:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        """
        初始化函数
        :param norm_epsilon: 方差加上极小的数，防止除以0的情况
        :param norm_decay: 在预测时计算moving average时的衰减率
        :param anchors_path: 先验框尺寸文件路径
        :param classes_path: 数据集类别文件路径
        :param pre_train: 是否使用预训练darknet53模型
        """
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.anchors = self._get_anchors()
        self.classes = self._get_class()

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
        return np.array(anchors).reshape(-1, 2)

    def _batch_normalization_layer(self, input_layer, name=None, training=True, norm_decay=0.99,
                                   norm_epsilon=1e-3):
        """
        对卷积层提取的feature map使用batch normalization
        :param input_layer: 上一层的输出
        :param name: bn层名字
        :param training: 是否为训练过程
        :param norm_decay: 在预测时计算moving average时的衰减率
        :param norm_epsilon: 方差加上极小的数，防止除以0
        :return bn_layers: batch normalization处理后的feature map
        """
        bn_layer = tf.layers.batch_normalization(
            inputs=input_layer,
            momentum=norm_decay,
            epsilon=norm_epsilon,
            center=True,
            scale=True,
            training=training,
            name=name
        )
        return tf.nn.leaky_relu(bn_layer, alpha=0.1)

    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias=False, strides=1):
        """
        卷积层
        :param inputs: 上一层的输出
        :param filters_num: 卷积核数量
        :param kernel_size: 卷积核尺寸
        :param name: 名称，标识此卷积层
        :param use_bias: 是否使用偏置项
        :param strides: 步长
        :return conv: 卷积后的feature map
        """
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filters_num,
            kernel_size=kernel_size,
            strides=[strides, strides],
            kernel_initializer=tf.glorot_uniform_initializer(),
            padding=('SAME' if strides == 1 else 'VALID'),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),  # L2正则化
            use_bias=use_bias,
            name=name
        )
        return conv

    def _Residual_block(self, inputs, filters_num, blocks_num, conv_index, training=True,
                        norm_decay=.99, norm_epsilon=1e-3):
        """
        残差卷积：
        Darknet的残差block，类似resnet的两层卷积结构，分别采用1x1和3x3的卷积核；
        进行一次3x3的卷积，得到layer，再进行一次1x1和一次3x3的卷积，并把这个结果加上layer作为最后的结果
        :param inputs: 上一层的输出
        :param filters_num: 卷积核数量
        :param blocks_num: block的数量
        :param conv_index: 为了方便加载预训练权重，统一命名序号
        :param training: 是否为训练过程
        :param norm_decay: 在预测时计算moving average时的衰减率
        :param norm_epsilon: 防除0
        :return: 经过残差网络处理后的结果
        """
        # 在输入feature map的长宽维度进行padding
        # [1, 0]表示在高度（height）维度上在前面（上方）填充1个单位，后面不填充。
        # [1, 0]表示在宽度（width）维度上在前面（左方）填充1个单位，后面不填充。
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        layer = self._conv2d_layer(inputs, filters_num, kernel_size=3, strides=2,
                                   name="conv2d_" + str(conv_index))
        layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index),
                                                training=training,
                                                norm_decay=norm_decay,
                                                norm_epsilon=norm_epsilon)
        conv_index += 1

        for _ in range(blocks_num):
            shortcut = layer
                                        # 处理layer
            layer = self._conv2d_layer(layer, filters_num // 2, kernel_size=1,
                                       strides=1, name="conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index),
                                                    training=training,
                                                    norm_decay=norm_decay,
                                                    norm_epsilon=norm_epsilon)
            conv_index += 1
            layer = self._conv2d_layer(layer, filters_num, kernel_size=3, strides=1,
                                       name="conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index),
                                                    training=training,
                                                    norm_decay=norm_decay,
                                                    norm_epsilon=norm_epsilon)
            conv_index += 1
            layer += shortcut
        # for循环后返回
        return layer, conv_index

    def _darknet53(self, inputs, conv_index, training=True, norm_decay=.99, norm_epsilon=1e-3):
        """
        构建yolo3使用的darknet53网络
        :param inputs: 输入
        :param conv_index: 卷积层序号，方便根据名字加载预训练权重
        :param training: 是否为训练过程
        :param norm_decay: 在预测时计算moving average时的衰减率
        :param norm_epsilon: 防除0
        :return:
        conv: 经过52层卷积计算之后的结果，输入图片为416x416x3，输出为13x13x1024
        route1: 返回第26层卷积计算结果52x52x256，供后续使用
        route2: 返回第43层卷积计算结果26x26x512，供后续使用
        """
        with tf.variable_scope('darknet53'):
            # 416x416x3 -> 416x416x32
            conv = self._conv2d_layer(inputs, filters_num=32, kernel_size=3, strides=1,
                                      name="conv2d_" + str(conv_index))
            conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index),
                                                   training=training,
                                                   norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1

            # 416x416x32 -> 208x208x64(residual里有一层pad)
            conv, conv_index = self._Residual_block(conv, 64, 1, conv_index=conv_index,
                                                    training=training,
                                                    norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            # 208x208x64 -> 104x104x128
            conv, conv_index = self._Residual_block(conv, 128, 2, conv_index=conv_index,
                                                    training=training,
                                                    norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            # 104x104x128 -> 52x52x256
            conv, conv_index = self._Residual_block(conv, 256, 8, conv_index=conv_index,
                                                    training=training,
                                                    norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            # route1: 52x52x256
            route1 = conv

            # 52x52x256 -> 26x26x512
            conv, conv_index = self._Residual_block(conv, 512, 8, conv_index=conv_index,
                                                    training=training,
                                                    norm_decay=norm_decay, norm_epsilon=norm_epsilon)

            # route2: 26x26x512
            route2 = conv

            # 26x26x512 -> 13x13x1024
            conv, conv_index = self._Residual_block(conv, 1024, 4, conv_index=conv_index,
                                                    training=training,
                                                    norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        return route1, route2, conv, conv_index

    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training=True,
                    norm_decay=0.99, norm_epsilon=1e-3):
        """
        yolo3在Darknet53提取的特征层基础上，又加了针对3种不同比例的feature map的block，来提高对小物体的检测率；
        输出两个网络结果：1、进行五次卷积后，用于下一次逆卷积的，卷积过程是1x1, 3x3, 1x1, 3x3, 1x1；
        2、进行5+2次卷积，作为一个特征层的，卷积过程是1x1, 3x3, 1x1, 3x3, 1x1, 3x3, 1x1
        :param inputs: 输入特征
        :param filters_num: 卷积核数量
        :param out_filters: 最后输出层的卷积核数量
        :param conv_index: 序号
        :param training: 是否为训练过程
        :param norm_decay: 计算moving average时的衰减率
        :param norm_epsilon: 防除0
        :return:
        route: 返回最后一层卷积的前一层结果
        conv: 返回最后一层卷积的结果
        conv_index: conv层的计数
        """
        conv = self._conv2d_layer(inputs, filters_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        route = conv

        conv = self._conv2d_layer(conv, filters_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, out_filters, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index), use_bias=True)
        conv_index += 1

        return route, conv, conv_index

    def yolo_inference(self, inputs, num_anchors, num_classes, training=True):
        """
        构建yolo3模型结构
        :param inputs: 模型的输入变量（darknet输出）
        :param num_anchors: 每个grid cell（网络单元格）负责检测的anchor数量
        :param num_classes: 类别数量
        :param training: 是否为训练模式
        :return: 返回三个特征层结果
        """
        # route1=52x52x256   route2=26x26x512  route3=13x13x1024
        conv2d_26, conv2d_43, conv, conv_index = self._darknet53(inputs, conv_index=1, training=training,
                                                                 norm_decay=self.norm_decay,
                                                                 norm_epsilon=self.norm_epsilon)
        with tf.variable_scope('yolo'):
            # ----------------------- #
            #     获取第一个特征层
            # ----------------------- #
            # conv2d_57 = 13x13x512, conv2d_59 = 13x13x255(3x(80+5))
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors * (num_classes + 5),
                                                                conv_index, training=training,
                                                                norm_decay=self.norm_decay,
                                                                norm_epsilon=self.norm_epsilon)
            # ----------------------- #
            #     获取第二个特征层
            # ----------------------- #
            # conv2d_60 = (conv2d_57 -> 13x13x256)
            conv2d_60 = self._conv2d_layer(conv2d_57, 256, kernel_size=1, strides=1,
                                           name="conv2d_" + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name="batch_normalization_" + str(conv_index),
                                                        training=training,
                                                        norm_decay=self.norm_decay,
                                                        norm_epsilon=self.norm_epsilon)
            conv_index += 1
            # up_sample_0 = 13x13x256->26x26x256
            up_sample_0 = tf.image.resize_nearest_neighbor(
                conv2d_60,
                [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]],
                name="up_sample_0"
            )
            # route0 = conv2d_43 + up_sample_0 = 26x26x(512+256)
            route0 = tf.concat([up_sample_0, conv2d_43], axis=-1, name="route_0")
            # conv2d_65 = 26x26x256, conv2d_67 = 26x26x255
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes + 5),
                                                                conv_index, training=training,
                                                                norm_decay=self.norm_decay,
                                                                norm_epsilon=self.norm_epsilon)
            # ----------------------- #
            #     获取第三个特征层
            # ----------------------- #
            # conv2d_68 = (conv2d_65 -> 26x26x128)
            conv2d_68 = self._conv2d_layer(conv2d_65, 128, kernel_size=1, strides=1,
                                           name="conv2d_" + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name="batch_normalization_" + str(conv_index),
                                                        training=training,
                                                        norm_decay=self.norm_decay,
                                                        norm_epsilon=self.norm_epsilon)
            conv_index += 1
            # up_sample_1 = 26x26x128 -> 52x52x128
            up_sample_1 = tf.image.resize_nearest_neighbor(
                conv2d_68,
                [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]],
                name="up_sample_1"
            )
            # route1 = conv2d_26 + up_sample_1 = 52x52x(256+128)
            route1 = tf.concat([up_sample_1, conv2d_26], axis=-1, name="route_1")
            # conv2d_75 = 52x52x255
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5),
                                               conv_index, training=training,
                                               norm_decay=self.norm_decay,
                                               norm_epsilon=self.norm_epsilon)
        return [conv2d_59, conv2d_67, conv2d_75]
