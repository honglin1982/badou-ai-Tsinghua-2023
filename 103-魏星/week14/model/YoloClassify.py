# _*_ coding : utf-8 _*_
# @Time : 2023/8/16 11:43
# @Author : weixing
# @FileName : YoloClassify
# @Project : cv

import torch.nn as nn
import torch
from DarkNet53 import DarkNet53, darknet53

'''
从特征获取预测结果的过程可以分为两个部分，分别是：
构建FPN特征金字塔进行加强特征提取。
对有效特征层进行预测结果。

1、特征提取部分一共生成三个特征层：(52,52,256)、(26,26,512)、(13,13,1024)
(13,13,1024)经过5轮卷积提取特征，用于上采样后与(26,26,512)结合(结合后为(26,26,768))，同时送入分类Head获得预测结果
(26,26,768)经过5轮卷积提取特征，用于上采样后与(52,52,256)结合(结合后为(52,52,384))，同时送入分类Head获得预测结果
(52,52,384)经过5轮卷积提取特征，送入分类Head获得预测结果

2、分类Head本质上是一次3x3卷积加上一次1x1卷积，3x3卷积的作用是特征整合，1x1卷积的作用是调整通道数。
其最后的维度为(52,52,N*(num_classes+1+4))、(26,26,N*(num_classes+1+4))、(13,13,N*(num_classes+1+4))
其中N为先验框个数，取值为3；num_classes为分类结果，不同数据集取不同的值，例如：VOC为20种，coco为80种；1：置信度，是否存在object，4：object的位置，x_offset、y_offset、h和w；
如果训练数据集为voc，则输出层结果shape为(13,13,75)，(26,26,75)，(52,52,75)
如果训练数据集为coco，则输出结果shape为(13,13,255)，(26,26,255)，(52,52,255)
'''


def convBnRelu(input_channel, out_channel, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(input_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(0.1)
    )


# 前5个用户特征提取，后2个用于分类预测
def get_final_layers(input_channel, mid_channel, out_channel):
    return nn.Sequential(
        convBnRelu(input_channel, mid_channel, kernel_size=1),
        convBnRelu(mid_channel, input_channel, kernel_size=3, padding=1),
        convBnRelu(input_channel, mid_channel, kernel_size=1),
        convBnRelu(mid_channel, input_channel, kernel_size=3, padding=1),
        convBnRelu(input_channel, mid_channel, kernel_size=1),

        convBnRelu(mid_channel, input_channel, kernel_size=3, padding=1),
        nn.Conv2d(input_channel, out_channel, kernel_size=1, stride=1, padding=0)
    )


def convUpSampling(input_channel, out_channel):
    return nn.Sequential(
        convBnRelu(input_channel, out_channel, kernel_size=1, stride=1, padding=0),
        nn.Upsample(scale_factor=2, mode='nearest')
    )


class YoloClassify(nn.Module):
    def __init__(self, num_classes):
        super(YoloClassify, self).__init__()

        self.backbone = darknet53()

        # (13,13,1024)->(13,13,512)->(13,13,255)
        self.final_layer_13 = get_final_layers(input_channel=1024, mid_channel=512, out_channel=3*(num_classes+4+1))
        # (13,13,512)->(26,26,256)
        self.layer_13_upSample = convUpSampling(input_channel=512, out_channel=256)

        # (26,26,768)->(26,26,256)->(26，26,255)
        self.final_layer_26 = get_final_layers(input_channel=512+256, mid_channel=256, out_channel=3*(num_classes+4+1))
        # (26,26,256)->(52,52,128)
        self.layer_26_upSample = convUpSampling(input_channel=256, out_channel=128)

        # (52,52,256+128)->(52,52,128)->(52,52,255)
        self.final_layer_52 = get_final_layers(input_channel=256+128, mid_channel=128, out_channel=3*(num_classes+4+1))

    def forward(self, x):
        out_52, out_26, out_13 = self.backbone(x)

        out_13_branch = self.final_layer_13[:5](out_13)         # [batch_size,13,13,1024] -> [batch_size,13,13,512]
        final_out_13 = self.final_layer_13[5:](out_13_branch)   # [batch_size,13,13,512] -> [batch_size,13,13,255]

        in_26 = self.layer_13_upSample(out_13_branch)           # [batch_size,13,13,512] -> [batch_size,26,26,256]
        out_26_cat = torch.cat([in_26, out_26], dim=1)          # [batch_size,26,26,256]+[batch_size,26,26,512] -> [batch_size,26,26,768]
        out_26_branch = self.final_layer_26[:5](out_26_cat)     # [batch_size,26,26,768] -> [batch_size,26,26,256]
        final_out_26 = self.final_layer_26[5:](out_26_branch)   # [batch_size,26,26,256] -> [batch_size,26,26,255]

        in_52 = self.layer_26_upSample(out_26_branch)           # [batch_size,26,26,256] -> [batch_size,52,52,128]
        out_52_cat = torch.cat([in_52, out_52], dim=1)          # [batch_size,52,52,256]+[batch_size,52,52,128] -> [batch_size,52,52,384]
        final_out_52 = self.final_layer_52(out_52_cat)          # [batch_size,52,52,384] -> [batch_size,52,52,128] -> [batch_size,52,52,255]

        return final_out_52, final_out_26, final_out_13

# model = YoloClassify(20)
# print(model)
