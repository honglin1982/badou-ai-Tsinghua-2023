# _*_ coding : utf-8 _*_
# @Time : 2023/8/4 14:37
# @Author : weixing
# @FileName : Faster-R-CNN
# @Project : cv

import torch
import torch.nn as nn

from nets.classifier import VGG16RoIHead
from nets.RPN import RegionProposalNetwork
from nets.vgg16 import get_vgg16

'''
1、图片resize到600*1000
2、conv layers：经backbone卷积网络提取特征（采用VGG16作为backbone，图片大小变为
    13个conv层：kernel_size=3，pad=1，stride=1；（经过卷积层，图片大小不变）
    +13个relu层：激活函数，不改变图片大小
    +4个pooling层：kernel_size=2，stride=2;pooling层会让输出图片是输入图片的1/2
    
    经过Conv layers，图片大小变成(M/16)*(N/16)，即：Feature Map就是(M/16)*(N/16)*512
3、RPN：主要分为两路：rpn_cls和rpn_bbox
    rpn_cls：对Anchor box内图像信息做二分类工作
    rpn_bbox：得到其Anchor box四个坐标信息（偏移量）
    
    比较通俗的讲就是，人为制定了9个不同大小的矩形框，又在1000x600的图像中，均匀的选取了60x40个位置，去放这9个矩形框，让矩形框在原图中游走。。。；
    所以，相当于在一个600x1000的图片中，选取了4060*9=21600个矩形框，这个数字很大，所以不能全部用，需要进一步做筛选（nms做筛选）。
    
    对预测框region proposal,进一步对预测框越界剔除和使用nms非最大值抑制，剔除掉重叠的框。
    RPN网络结束，经过此网络筛选，anchor box由原来的21600变为300个。
    数据输入到ROI Pooling层进行进一步的分类和定位。
4、ROI Pooling：输入的是RPN层产生的region proposal（假定有300个region proposal box）和VGG16最后一层产生的特征图(60x40x512-d)，
    遍历每个region proposal，将其坐标值缩小16倍，这样就可以将在原图(1000x600)基础上产生的region proposal映射到60x40的特征图上，
    从而将在feature map上确定一个区域(定义为RB*)。在feature map上确定的区域RB*，根据参数pooled_w:7,pooled_h:7，将这个RB区域划分为7x7，
    即49个相同大小的小区域，对于小区域，使用max pooling方式从中选取最大的像素点作为输出，这样，就形成了一个7x7的feature map。
    
    通俗点讲，就是输入一个60x40的特征图，共512维；然后再特征图中画300个矩形框(也就是region proposal)，对每一个矩形框通过roi_pooling，
    变成7x7的特征图，也就是7x7x512x300。
    
5、全连接层：
    经过roi pooling层之后，batch_size=300，proposal feature map的大小是7x7x512-d，对特征图进行全连接，最后同样利用softmax Loss和L1 loss完成分类和定位。
    通过full connect层与softmax计算每个region proposal具体属于哪个类别，输出cls_prob概率向量；
    同时再次利用bbox regression获得每个region proposal的位置偏移量bbox_pred，用于回归获得更加精确的目标检测框。
    
'''


class FasterRCNN(nn.Module):
    def __init__(self, num_classes,
                 mode="training",
                 feat_stride=16,
                 anchor_scales=[8, 16, 32],
                 ratios=[0.5, 1, 2],
                 backbone='vgg',
                 pretrained=False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        if backbone == 'vgg':
            self.extractor, classifier = get_vgg16(pretrained)
            # ---------------------------------#
            #   构建建议框网络
            # ---------------------------------#
            self.rpn = RegionProposalNetwork(
                512, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode=mode
            )
            # ---------------------------------#
            #   构建分类器网络
            # ---------------------------------#
            self.head = VGG16RoIHead(
                n_class=num_classes + 1,
                roi_size=7,
                spatial_scale=1,
                classifier=classifier
            )

    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":
            # ---------------------------------#
            #   计算输入图片的大小
            # ---------------------------------#
            img_size = x.shape[2:]
            # ---------------------------------#
            #   利用主干网络提取特征
            # ---------------------------------#
            base_feature = self.extractor.forward(x)

            # ---------------------------------#
            #   获得建议框
            # ---------------------------------#
            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
            # ---------------------------------------#
            #   获得classifier的分类结果和回归结果
            # ---------------------------------------#
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            # ---------------------------------#
            #   利用主干网络提取特征
            # ---------------------------------#
            base_feature = self.extractor.forward(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            # ---------------------------------#
            #   获得建议框
            # ---------------------------------#
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            # ---------------------------------------#
            #   获得classifier的分类结果和回归结果
            # ---------------------------------------#
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()




