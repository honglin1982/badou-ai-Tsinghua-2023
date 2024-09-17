# _*_ coding : utf-8 _*_
# @Time : 2023/8/17 17:12
# @Author : weixing
# @FileName : RNet
# @Project : cv

import torch.nn as nn

'''
全称为Refine Network，其基本的构造是一个卷积神经网络，
相对于第一层的P-Net来说，增加了一个全连接层，因此对于输入数据的筛选会更加严格。

在图片经过P-Net后，会留下许多预测窗口，我们将所有的预测窗口送入R-Net，这个网络会滤除大量效果比较差的候选框，
最后对选定的候选框进行Bounding-Box Regression和NMS进一步优化预测结果。
'''


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.fpn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3),  # (24,24,3)->(22,22,28)
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True),  # (22,22,28) -> (11,11,28)

            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3),  # (11,11,28)->(9,9,48)
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True),  # (9,9,48)->(4,4,48)

            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2),  # (4,4,48)->(3,3,64)
            nn.PReLU()
        )

        self.flatten = nn.Flatten()  # (3,3,64) -> (576)
        self.fc = nn.Linear(in_features=3 * 3 * 64, out_features=128)  # (576)->(128)

        self.face_class_fc = nn.Linear(128, 2)  # (128)->(2)
        self.bbox_reg_fc = nn.Linear(128, 4)  # (128)->(4)
        self.facial_landmark_fc = nn.Linear(128, 10)  # (128)->(10)

    def forward(self, x):
        x = self.fpn(x)
        x = self.flatten(x)
        x = self.fc(x)

        class_out = self.face_class_fc(x)

        bbox_out = self.bbox_reg_fc(x)

        landmark_out = self.facial_landmark_fc(x)

        return class_out, bbox_out, landmark_out
