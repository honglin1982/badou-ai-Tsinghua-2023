import numpy as np

class Config(object):

    NAME = None
    GPU_COUNT = 1          # 使用 GPU 数量（如果只用CPU，可以设置为1）
    IMAGES_PER_GPU = 2     # 每个 GPU 训练图片的个数(12GB 的 GPU 一般可以处理两张 1024 * 1024）
    STEPS_PER_EPOCH = 1000 # 每个世代的训练迭代次数
    VALIDATION_STEPS = 50  # 每个世代的训练迭代次数中的有效次数（次数越大准确率会提高，但训练会更好耗时）
    BACKBONE = "resnet101" # 网络结构的主干
    COMPUTE_BACKBONE_SHAPE = None #只有调用 backbone 时有效，用于计算每一层 特征金字塔网络的 shape
    BACKBONE_STRIDES = [4, 8, 16, 32, 64] #每层特征金字塔网络 的步长（基于 Resnet101）
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024     #分类图像部分 全连接层的size
    TOP_DOWN_PYRAMID_SIZE = 256           #被用于特征金字塔的自上而下的层的 size
    NUM_CLASSES = 1                       #分类类别数
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512) #候选区域网络的anchor大小
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]             # anchor的宽高比（1表示正方形的anchor，0.5表示宽的anchor）
    # anchor步长（1表示为anchor是为 每个 位于feature map 中 cell 生成的，
    RPN_ANCHOR_STRIDE = 1                   #2表示anchor是为了每个其他cell生成的
    RPN_NMS_THRESHOLD = 0.7                 # NMS 阈值 过滤 候选区域框
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256       # 每个图片有多少anchors 用于 RPN 训练
    PRE_NMS_LIMIT = 6000                    # 在tf.nn.top_k和非最大抑制之前保持的 ROI 层的 候选框数量

    #经过 NMS 后保留的 候选框数量（训练和推理）
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    #如果启用，会将实例掩码的大小调整为较小的大小以减少内存负载。建议使用高分辨率图像时使用。
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56) # 最小掩码的高和宽

    #修改输入图像的大小
    IMAGE_RESIZE_MODE = "square" #一般为 square，调整并通过填充0得到一个正方形图像 在大多数情况下它应该运行良好
    IMAGE_MIN_DIM = 800          # 在此模式下，图像被缩放向上使得小边 = IMAGE_MIN_DIM，但确保 缩放不会使长边> IMAGE_MAX_DIM
    IMAGE_MAX_DIM = 1024         # 然后用零填充以使其成为正方形，以便可以放置多个图像到一个批次中
    #其他模式 None：返回不做处理的图像
    #       pad64：pad的宽度和高度用零表示，使它们成为 64 的倍数，如果IMAGE_MIN_DIM或IMAGE_MIN_SCALE不是 None
    #             则它会在填充之前向上扩展。在此模式下IMAGE_MAX_DIM将被忽略。 需要 64 的倍数来确保功能的平滑缩放
    #             映射 FPN 金字塔的 6 个级别 （2**6=64）
    #       crop：从图像中选取随机裁剪。首先，根据IMAGE_MIN_DIM和IMAGE_MIN_SCALE缩放图像，
    #             然后随机选择尺寸 IMAGE_MIN_DIM x IMAGE_MIN_DIM。只能在训练中使用。
    #             在此模式下不使用IMAGE_MAX_DIM
    # 最小缩放比率。检查MIN_IMAGE_DIM后，可以强制进一步扩展。例如
    # 如果设置为 2，则图像将放大到宽度和高度的两倍或更多，即使MIN_IMAGE_DIM不要求它。
    # 但是，在“square”模式下，它可以被IMAGE_MAX_DIM否决
    IMAGE_MIN_SCALE = 0
    IMAGE_CHANNEL_COUNT = 3 #图像通道数
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9]) #图像均值
    # 每张图像要喂给 分类器 / 掩模的ROI数量
    # Mask RCNN论文使用512，但RPN通常不会生成
    # 足够的 positive 候选框来保证 pos ：neg
    # 比例为1：3。可以通过调整NMS阈值来增加候选框数量
    TRAIN_ROIS_PER_IMAGE = 200
    # positive ROIS 用于训练 分类/掩膜 训练的百分比
    ROI_POSITIVE_RATIO = 0.33

    # Pool 所需 候选框数
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    #输出掩膜大小（改变这个神经网络中的掩膜分支也要改变）
    MASK_SHAPE = [28, 28]
    #在一个图像中使用的真实例的最大数量
    MAX_GT_INSTANCES = 100
    #RPN 和最终检测的边界框细化标准偏差
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    #最终检测的最大数量
    DETECTION_MAX_INSTANCES = 100
    #检测实例的最小置信度
    DETECTION_MIN_CONFIDENCE = 0.7
    #用于检测的 NMS 阈值
    DETECTION_NMS_THRESHOLD = 0.3
    #学习率和动量
    #Mask RCNN 论文使用 lr=0.02，但在 TensorFlow 上它会导致权重爆炸
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    #权重衰减
    WEIGHT_DECAY = 0.0001

    # 权重 loss
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    #使用 RPN ROI 或外部生成的 ROI 进行训练
    #在大多数情况下保持此 True
    #如果要根据代码生成的 ROI 而不是 RPN 的 ROI 来训练分支，请设置为 False。例如，调试分类器而无需训练 RPN。
    USE_RPN_ROIS = True
    # 训练或冻结批量归一化层
    # None：训练 BN 层。这是正常模式
    # False：冻结 BN 图层。使用小批量时很好
    # True：（不使用）即使在预测时也设置训练模式中的图层（Set layer in training mode even when predicting）
    TRAIN_BN = False #默认设置为 False(因为批量一般都很小)
    #渐变范数裁剪
    GRADIENT_CLIP_NORM = 5.0


    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                self.IMAGE_CHANNEL_COUNT])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
