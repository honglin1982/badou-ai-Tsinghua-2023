# _*_ coding : utf-8 _*_
# @Time : 2023/8/17 16:41
# @Author : weixing
# @FileName : PNet
# @Project : cv

import torch.nn as nn
import torch

"""
PNet全称为Proposal Network，其基本的构造是一个全卷积网络，
P-Net是一个人脸区域的区域建议网络，该网络的将特征输入结果三个卷积层之后，
通过一个人脸分类器判断该区域是否是人脸，同时使用边框回归。

"""


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.fpn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3),  # (12,12,3)->(10,10,10)
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),     # (10,10,10) -> (5,5,10)

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3),  # (5,5,10)->(3,3,16)
            nn.PReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),  # (3,3,16)->(1,1,32)
            nn.PReLU()
        )

        self.face_class_conv = nn.Conv2d(32, 2, kernel_size=1)   # (1,1,32)->(1,1,2)
        self.bbox_reg_conv = nn.Conv2d(32, 4, kernel_size=1)  # (1,1,32)->(1,1,4)
        self.facial_landmark_conv = nn.Conv2d(32, 10, kernel_size=1)  # (1,1,32)->(1,1,10)

    def forward(self, x):
        x = self.fpn(x)

        # 是否包含物体
        class_out = self.face_class_conv(x)
        class_out = torch.squeeze(class_out, dim=0)
        class_out = torch.squeeze(class_out, dim=0)

        # 人脸待选框
        bbox_out = self.bbox_reg_conv(x)
        bbox_out = torch.squeeze(bbox_out, dim=0)
        bbox_out = torch.squeeze(bbox_out, dim=0)

        # 5个关键点
        landmark_out = self.facial_landmark_conv(x)
        landmark_out = torch.squeeze(landmark_out, dim=0)
        landmark_out = torch.squeeze(landmark_out, dim=0)

        return class_out, bbox_out, landmark_out
