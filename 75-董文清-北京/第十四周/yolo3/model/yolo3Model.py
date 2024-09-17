import os
import numpy as np
import tensorflow as tf


class yolo:

    def __init__(self, normEpsilon, normDecay, anchorsPath, classesPath, preTrain):
        """
           Introduction
           ------------
               初始化函数
           Parameters
           ----------
               normDecay: 在预测时计算moving average时的衰减率
               normEpsilon: 方差加上极小的数，防止除以0的情况
               anchorsPath: yolo anchor 文件路径
               classesPath: 数据集类别对应文件
               preTrain: 是否使用预训练darknet53模型
        """
        self.normEpsilon = normEpsilon
        self.normDecay = normDecay
        self.anchorsPath = anchorsPath
        self.classesPath = classesPath
        self.preTrain = preTrain
        self.anchors = self._getAnchors()
        self.classes = self._getClass()

    # 获取anchors
    def _getAnchors(self):

        anchorsPath = os.path.expanduser(self.anchorsPath)
        with open(anchorsPath) as f: anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    # 获取类别名
    def _getClass(self):

        classesPath = os.path.expanduser(self.classesPath)
        with open(classesPath) as f: classNames = f.readlines()
        classNames = [c.strip() for c in classNames]
        return classNames

    # 生成层 l2正则化
    def batchNormalizationLayer(self, inputLayer, name=None, training=True, normDecay=0.99, normEpsilon=1e-3):
        '''
        Introduction
        ------------
            对卷积层提取的feature map使用batch normalization
        Parameters
        ----------
            inputLayer: 输入的四维tensor
            name: batchnorm层的名字
            trainging: 是否为训练过程
            normDecay: 在预测时计算moving average时的衰减率
            normEpsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            bnLayer: batch normalization处理之后的feature map
        '''
        bnLayer = tf.layers.batch_normalization(inputs=inputLayer, momentum=normDecay, epsilon=normEpsilon,
                                                center=True, scale=True, training=training, name=name)
        return tf.nn.leaky_relu(bnLayer, alpha=0.1)

    # 生成卷积层
    def _conv2dLayer(self, inputs, filtersNum, kernelSize, name, useBias=False, strides=1):
        """
        Introduction
        ------------
            使用tf.layers.conv2d减少权重和偏置矩阵初始化过程，以及卷积后加上偏置项的操作
            经过卷积之后需要进行batch norm，最后使用leaky ReLU激活函数
            根据卷积时的步长，如果卷积的步长为2，则对图像进行降采样
            比如，输入图片的大小为416*416，卷积核大小为3，若stride为2时，（416 - 3 + 2）/ 2 + 1， 计算结果为208，相当于做了池化层处理
            因此需要对stride大于1的时候，先进行一个padding操作, 采用四周都padding一维代替'same'方式
        Returns
        -------
            conv: 卷积之后的feature map
        """
        conv = tf.layers.conv2d(inputs=inputs, filters=filtersNum, kernel_size=kernelSize,
                                strides=[strides, strides], kernel_initializer=tf.glorot_uniform_initializer(),
                                padding=('SAME' if strides == 1 else 'VALID'), kernel_regularizer=
                                tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=useBias, name=name)
        return conv

    # 残差卷积 进行一次 3 x 3 卷积，然后保存卷积layer，
    # 再进行一次 1 x 1 卷积和一次 3 x 3卷积，并把结果加上layer作为返回的最后结果
    def _ResidualBlock(self, inputs, filtersNum, blocksNum, convIndex, training=True, normDecay=0.99, normEpsilon=1e-3):
        """
        Introduction
        ------------
            Darknet的残差block，类似resnet的两层卷积结构，分别采用1x1和3x3的卷积核，使用1x1是为了减少channel的维度
        Parameters
        ----------
            inputs: 输入变量
            filtersNum: 卷积核数量
            trainging: 是否为训练过程
            blocksNum: block的数量
            convIndex: 为了方便加载预训练权重，统一命名序号
        Returns
        -------
            layer, convIndex: 经过残差网络处理后的结果
        """
        # 填充padding 在
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        layer = self._conv2dLayer(inputs, filtersNum, kernelSize=3, strides=2, name='conv2d_' + str(convIndex))
        layer = self.batchNormalizationLayer(layer, name='batch_normalization_' + str(convIndex), training=training,
                                             normDecay=normDecay, normEpsilon=normEpsilon)
        convIndex += 1
        for _ in range(blocksNum):
            shortcut = layer
            layer = self._conv2dLayer(layer, filtersNum // 2, kernelSize=1, strides=1, name='conv2d_' + str(convIndex))
            layer = self.batchNormalizationLayer(layer, name='batch_normalization_' + str(convIndex), training=training,
                                                 normDecay=normDecay, normEpsilon=normEpsilon)
            convIndex += 1
            layer = self._conv2dLayer(layer, filtersNum, kernelSize=3, strides=1, name='conv2d_' + str(convIndex))
            layer = self.batchNormalizationLayer(layer, name='batch_normalization_' + str(convIndex), training=training,
                                                 normDecay=normDecay, normEpsilon=normEpsilon)
            convIndex += 1
            layer += shortcut
        return layer, convIndex

    # 生成 darkernet53
    def _darkernet53(self, inputs, convIndex, training=True, normDecay=0.99, normEpsilon=1e-3):
        """
        Introduction
        ------------
            构建yolo3使用的darknet53网络结构
        Parameters
        ----------
            inputs: 模型输入变量
            convIndex: 卷积层数序号，方便根据名字加载预训练权重
            training: 是否为训练
        Returns
        -------
            conv: 经过52层卷积计算之后的结果, 输入图片为416x416x3，则此时输出的结果shape为13x13x1024
            route1: 返回第26层卷积计算结果52x52x256, 供后续使用
            route2: 返回第43层卷积计算结果26x26x512, 供后续使用
            conv_index: 卷积层计数，方便在加载预训练模型时使用
        """
        with tf.variable_scope('darknet53'):
            # 416,416,3 -> 416,416,32
            conv = self._conv2dLayer(inputs, filtersNum=32, kernelSize=3, strides=1, name='conv2d_' + str(convIndex))
            conv = self.batchNormalizationLayer(conv, name='batch_normalization_' + str(convIndex), training=training,
                                                normDecay=normDecay, normEpsilon=normEpsilon)
            convIndex += 1
            # 416,416,32 -> 208,208,64
            conv, convIndex = self._ResidualBlock(conv, convIndex=convIndex, filtersNum=64, blocksNum=1,
                                                  training=training, normDecay=normDecay, normEpsilon=normEpsilon)
            # 208,208,64 -> 104,104,128
            conv, convIndex = self._ResidualBlock(conv, convIndex=convIndex, filtersNum=128, blocksNum=2,
                                                  training=training, normDecay=normDecay, normEpsilon=normEpsilon)
            # 104,104,128 -> 52,52,256
            conv, convIndex = self._ResidualBlock(conv, convIndex=convIndex, filtersNum=256, blocksNum=8,
                                                  training=training, normDecay=normDecay, normEpsilon=normEpsilon)
            route1 = conv
            # 52,52,256 -> 26,26,512
            conv, convIndex = self._ResidualBlock(conv, convIndex=convIndex, filtersNum=512, blocksNum=8,
                                                  training=training, normDecay=normDecay, normEpsilon=normEpsilon)
            route2 = conv
            # 26,26,512 -> 13,13,1024
            conv, convIndex = self._ResidualBlock(conv, convIndex=convIndex, filtersNum=1024, blocksNum=4,
                                                  training=training, normDecay=normDecay, normEpsilon=normEpsilon)
        return route1, route2, conv, convIndex

    # 输出两个网络结果
    # 第一个是进行5次卷积后，用于下一次逆卷积的，卷积过程是1X1，3X3，1X1，3X3，1X1
    # 第二个是进行5+2次卷积，作为一个特征层的，卷积过程是1X1，3X3，1X1，3X3，1X1，3X3，1X1
    def _yoloBlock(self, inputs, filtersNum, outFilters, convIndex, training=True, normDecay=0.99, normEpsilon=1e-3):
        """
        Introduction
        ------------
            yolo3在Darknet53提取的特征层基础上，又加了针对3种不同比例的feature map的block，这样来提高对小物体的检测率
        Parameters
        ----------
            inputs: 输入特征
            filtersNum: 卷积核数量
            outFilters: 最后输出层的卷积核数量
        Returns
        -------
            route: 返回最后一层卷积的前一层结果
            conv: 返回最后一层卷积的结果
            conv_index: conv层计数
        """
        conv = self._conv2dLayer(inputs, filtersNum=filtersNum, kernelSize=1, strides=1,
                                 name='conv2d_' + str(convIndex))
        conv = self.batchNormalizationLayer(conv, name='batch_normalization_' + str(convIndex), training=training,
                                            normDecay=normDecay, normEpsilon=normEpsilon)
        for i in range(3):

            convIndex += 1
            conv = self._conv2dLayer(conv, filtersNum=filtersNum * 2, kernelSize=3, strides=1,
                                     name='conv2d_' + str(convIndex))
            conv = self.batchNormalizationLayer(conv, name='batch_normalization_' + str(convIndex), training=training,
                                                normDecay=normDecay, normEpsilon=normEpsilon)

            convIndex += 1
            if i == 2: break
            conv = self._conv2dLayer(conv, filtersNum=filtersNum, kernelSize=1, strides=1,
                                     name='conv2d_' + str(convIndex))
            conv = self.batchNormalizationLayer(conv, name='batch_normalization_' + str(convIndex), training=training,
                                                normDecay=normDecay, normEpsilon=normEpsilon)
            if i == 1: route = conv

        conv = self._conv2dLayer(conv, filtersNum=outFilters, kernelSize=1, strides=1,
                                 name='conv2d' + str(convIndex), useBias=True)
        convIndex += 1
        return route, conv, convIndex

    # 返回三个特征层内容
    def yoloInference(self, inputs, numAnchors, numClasses, training=True):
        """
        Introduction
        ------------
            构建yolo模型结构
        Parameters
        ----------
            inputs: 模型的输入变量
            numAnchors: 每个grid cell负责检测的anchor数量
            numClasses: 类别数量
            training: 是否为训练模式
        """
        convIndex = 1
        # route1 = 52,52,256、route2 = 26,26,512、route3 = 13,13,1024
        conv2d26, conv2d43, conv, convIndex = self._darkernet53(inputs, convIndex, training=training,
                                                                normDecay=self.normDecay, normEpsilon=self.normEpsilon)
        with tf.variable_scope('yolo'):
            # 获取第一个特征层
            # conv2d_57 = 13,13,512，conv2d_59 = 13,13,255(3x(80+5))
            conv2d57, conv2d59, convIndex = self._yoloBlock(conv, 512, numAnchors * (numClasses + 5),
                                                            convIndex=convIndex,
                                                            training=training, normDecay=self.normDecay,
                                                            normEpsilon=self.normEpsilon)
            # 获取第二个特征层
            conv2d60 = self._conv2dLayer(conv2d57, filtersNum=256, kernelSize=1, strides=1,
                                         name='conv2d_' + str(convIndex))
            conv2d60 = self.batchNormalizationLayer(conv2d60, name='batch_normalization_' + str(convIndex),
                                                    training=training,
                                                    normDecay=self.normDecay, normEpsilon=self.normEpsilon)
            convIndex += 1
            # unSample0 = 26,26,256 上采样
            unSample0 = tf.image.resize_bilinear(conv2d60, [2 * tf.shape(conv2d60)[1], 2 * tf.shape(conv2d60)[1]],
                                                 name='upSample0')
            # route0 = 26,26,768
            route0 = tf.concat([unSample0, conv2d43], axis=-1, name='route0')
            # conv2d_65 = 52,52,256，conv2d_67 = 26,26,255
            conv2d65, conv2d67, convIndex = self._yoloBlock(route0, 256, numAnchors * (numClasses + 5),
                                                            convIndex=convIndex,
                                                            training=training, normDecay=self.normDecay,
                                                            normEpsilon=self.normEpsilon)
            # 获取第三个特征层
            conv2d68 = self._conv2dLayer(conv2d65, filtersNum=128, kernelSize=1, strides=1,
                                         name='conv2d_' + str(convIndex))
            conv2d68 = self.batchNormalizationLayer(conv2d68, name='batch_normalization_' + str(convIndex),
                                                    training=training,
                                                    normDecay=self.normDecay, normEpsilon=self.normEpsilon)
            convIndex += 1
            # unSample_1 = 52,52,128
            unSample1 = tf.image.resize_bilinear(conv2d68, [2 * tf.shape(conv2d68)[1], 2 * tf.shape(conv2d68)[1]],
                                                 name='upSample1')
            # route1= 52,52,384
            route1 = tf.concat([unSample1, conv2d26], axis=-1, name='route1')
            _, conv2d75, _ = self._yoloBlock(route1, 128, numAnchors * (numClasses + 5), convIndex=convIndex,
                                             training=training, normDecay=self.normDecay, normEpsilon=self.normEpsilon)
        return [conv2d59, conv2d67, conv2d75]
