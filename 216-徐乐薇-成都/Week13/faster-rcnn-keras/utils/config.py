from keras import backend as K

# 本文件用于配置一些超参数
class Config:

    def __init__(self):
        self.anchor_box_scales = [128, 256, 512]            # anchor的大小，分别对应三种尺度
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]   # anchor的比例，分别对应三种尺度
        self.rpn_stride = 16                                # 下采样倍数
        self.num_rois = 32                                  # 一张图片中的roi数量，不超过32个框
        self.verbose = True                                 # 是否打印网络结构
        self.model_path = "logs/model.h5"                   # 训练好的模型保存的位置
        self.rpn_min_overlap = 0.3                          # rpn网络的正负样本比例
        self.rpn_max_overlap = 0.7                          # rpn网络的正负样本比例
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
        