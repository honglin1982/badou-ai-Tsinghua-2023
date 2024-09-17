import numpy as np


class Config(object):
    """
    基本配置类。对于自定义配置，请创建从该子类继承并重写属性的子类
    这需要改变。
    """
    # 命名配置
    NAME = None

    # 要使用的GPU数量，当仅使用CPU时，需要将其设置为1
    GPU_COUNT = 1

    # 在每个GPU上训练的图像数。12GB GPU通常可以
    # 处理2个1024x1024px的图像。
    # 根据GPU内存和图像大小进行调整。使用最高
    # GPU可以处理的数字以获得最佳性能。
    IMAGES_PER_GPU = 2

    # 每代的训练步骤数
    # 这不需要与训练集的大小相匹配。Tensorboard
    # 更新保存在每个epoch的末尾，因此将其设置为
    # 数字越小，TensorBoard的更新频率就越高。
    # 验证统计数据也会在每代结束时进行计算
    # 可能需要一段时间，所以不要把这个设定得太小以避免支出
    # 在验证统计数据上花了很多时间。
    STEPS_PER_EPOCH = 1000

    # 在每个训练时期结束时要运行的验证步骤数。
    # 更大的数字可以提高验证统计数据的准确性，但速度会变慢
    # 结束训练。
    VALIDATION_STEPS = 50

    # 骨干网络架构
    # 支持的值为：resnet50、resnet101。
    # 您还可以提供一个应具有签名的可调用
    # 的。如果你这样做，你需要提供一个可调用的
    # COMPUTE_BACKBONE_SHAPE
    BACKBONE = 'resnet101'

    # 只有当您向BACKBONE提供可调用时才有用。应计算
    # FPN金字塔的每一层的形状。
    # 请参阅model.compute_backline_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # FPN金字塔每一层的步伐。这些值
    # 基于Resnet101骨干网。
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # 分类图中完全连接层的大小
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # 用于构建要素棱锥体的自上而下的图层的大小
    TOP_DOWN_PYRAMID_SIZE = 256

    # 分类类别数量（包括背景）
    NUM_CLASSES = 1

    # 方形锚定边的长度（像素）
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # 每个单元的锚固件比率（宽度/高度）
    # 值1表示方形锚点，0.5表示宽锚点
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # 锚定步幅
    # 如果为1，则为主干特征图中的每个单元创建锚点。
    # 如果为2，则每隔一个单元创建一个锚点，依此类推。
    RPN_ANCHOR_STRIDE = 1

    # 用于过滤RPN建议的非最大抑制阈值。
    # 你可以在训练中增加这一点，以产生更多的推动力。
    RPN_NMS_THRESHOLD = 0.7

    # 每个图像要用于RPN训练的锚数量
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # 在tf.nn.top_k之后和非最大抑制之前保留的ROI
    PRE_NMS_LIMIT = 6000

    # 非最大抑制（训练和推理）后保留的ROI
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # 如果启用，则将实例掩码调整为较小的大小以减小
    # 内存负载。建议在使用高分辨率图像时使用。
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    # 输入图像大小调整
    # 通常，使用“方形”调整大小模式进行训练和预测
    # 而且它在大多数情况下应该都能很好地工作。在该模式中，图像被缩放
    # 向上，使小边为=IMAGE_MIN_DIM，但确保
    # 缩放不会使长边>IMAGE_MAX_DIM。然后图像是
    # 填充0使其成为正方形，以便可以放置多个图像
    # 一批。
    # 可用的调整大小模式：
    # none：没有调整大小或填充。返回未更改的图像。
    # square：调整大小并用零填充以获得方形图像
    # 大小为[max_dim，max_dim]。
    # pad64：用零填充宽度和高度，使其成为64的倍数。
    # 如果IMAGE_MIN_DIM或IMAGE_MING_SCALE不为“无”，则缩放
    # 填充前向上。IMAGE_MAX_DIM在此模式下被忽略。
    # 需要64的倍数来确保特征的平滑缩放
    # 映射FPN金字塔的6个级别（2**6=64）。
    # crop：从图像中拾取随机作物。首先，基于缩放图像
    # 在IMAGE_MIN_DIM和IMAGE_MING_SCALE上，然后随机选取
    # 尺寸IMAGE_MIN_DIM x IMAGE_MIN _DIM。只能用于培训。
    # IMAGE_MAX_DIM不在此模式中使用。
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # 最小缩放比。在MIN_IMAGE_DIM之后检查，可以进一步强制
    # 放大。例如，如果设置为2，则图像会放大一倍
    # 宽度和高度，甚至更多，即使MIN_IMAGE_DIM不需要。
    # 但是，在“方形”模式下，它可以被IMAGE_MAX_DIM否决。
    IMAGE_MIN_SCALE = 0

    # 每个图像的颜色通道数。RGB=3，灰度=1，RGB-D=4
    # 更改此项需要对代码进行其他更改。查看WIKI了解更多信息
    # 详细信息：https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # 图像平均值（RGB）
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # 每个图像要馈送到分类器/掩模头的ROI数量
    # Mask RCNN论文使用512，但RPN通常不会生成
    # 足够多的积极建议来填补这一空白并保持积极：消极
    # 比例为1:3。您可以通过调整
    # RPN NMS阈值。
    TRAIN_ROIS_PER_IMAGE = 200

    # 用于训练分类器/掩模头的阳性ROI百分比
    ROI_POSITIVE_RATIO = 0.33

    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # 输出掩码的形状
    # 要改变这一点，还需要改变神经网络掩码分支
    MASK_SHAPE = [28, 28]

    # 在一个图像中使用的最大地面实况实例数
    MAX_GT_INSTANCES = 100

    # RPN和最终检测的边界框细化标准偏差。
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # 最终检测的最大次数
    DETECTION_MAX_INSTANCES = 100

    # 接受检测到的实例的最小概率值
    # 跳过低于此阈值的ROI
    DETECTION_MIN_CONFIDENCE = 0.7

    # 检测的非最大抑制阈值
    DETECTION_NMS_THRESHOLD = 0.3

    # 学习率和动力
    # Mask RCNN论文使用lr=0.02，但在TensorFlow上它会导致
    # 要爆炸的重量。可能是由于优化器的差异
    # 执行。
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # 权重衰减正则化
    WEIGHT_DECAY = 0.0001

    # 损失权重以实现更精确的优化。
    # 可用于R-CNN训练设置。
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # 使用RPN ROI或外部生成的ROI进行培训
    # 在大多数情况下保持这一点。如果要训练，请设置为False
    # 头部在由代码生成的ROI上分支，而不是来自
    # RPN。例如，无需调试分类器头
    # 训练RPN。
    USE_RPN_ROIS = True

    # 训练或冻结批处理规范化层
    # None：训练BN层。这是正常模式
    # False：冻结BN层。使用小批量时效果良好
    # True：（不要使用）。即使在预测时也将图层设置为训练模式
    TRAIN_BN = False

    # 渐变范数剪裁
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        """设置计算属性的值"""
        # 有效批量大小
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # 输入图片大小
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                                         self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                                         self.IMAGE_CHANNEL_COUNT])

        # 图象元数据长度
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """显示配置信息"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")