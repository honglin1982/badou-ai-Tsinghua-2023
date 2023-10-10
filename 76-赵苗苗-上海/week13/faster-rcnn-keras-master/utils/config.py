from keras import backend as K
#定义了名为Config的类，提供了一组默认的参数值，用于配置目标检测算法的超参数和模型路径
class Config:
    def __init__(self):
        self.anchor_box_scales=[128,256,512]  #列表，用于生成锚框的尺寸
        self.anchor_box_ratios=[[1,1],[1,2],[2,1]]  #二维列表，用于生成锚框的宽高比
        self.rpn_stride=16  #RPN的步长
        self.num_rois=32    #训练样本中选择的区域数量，
        self.verbose=True
        self.rpn_min_overlap=0.3  #正样本的IoU阈值
        self.rpn_max_overlap=0.7  #负样本的IoU阈值
        self.classifier_min_overlap=0.1  #分类的候选区的IoU阈值
        self.classifier_max_overlap=0.5  #回归的候选区的IoU阈值
        self.classifier_regr_std=[8.0,8.0,4.0,4.0]   #回归损失计算的标准差
        self.model_path="logs/model.h5"   #保存训练模型的路径