# _*_ coding : utf-8 _*_
# @Time : 2023/7/24 17:48
# @Author : weixing
# @FileName : network
# @Project : cv

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()

        # 提取特征
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # input [224,224,3] output [224,224,64]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # input [224,224,64] output [224,224,64]
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),  # input [224,224,64] output [112,112,64]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # input [112,112,64] output [112,112,128]
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # input [112,112,128] output [112,112,128]
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),  # input [112,112,128] output [56,56,128]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # input [56,56,128] output [56,56,256]
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # input [56,56,256] output [56,56,256]
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # input [56,56,256] output [56,56,256]
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),  # input [56,56,256] output [28,28,256]

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # input [28,28,256] output [28,28,512]
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # input [28,28,512] output [28,28,512]
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # input [28,28,512] output [28,28,512]
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),  # input [28,28,512] output [14,14,512]

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # input [14,14,512] output [14,14,512]
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # input [14,14,512] output [14,14,512]
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # input [14,14,512] output [14,14,512]
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)  # input [14,14,512] output [7,7,512]
        )

        # 分类,用卷积代理全连接
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0),  # input [7,7,512] output [1,1,4096]
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Conv2d(4096, 4096, kernel_size=3, stride=1, padding=1),  # input [1,1,4096] output [1,1,4096]
            nn.ReLU(inplace=True),

            nn.Conv2d(4096, num_classes, kernel_size=3, stride=1, padding=1)  # input [1,1,4096] output [1,1,num_classes]
        )

    def forward(self, x):  # 正向传递的函数
        x = self.features(x)

        x = self.classifier(x)

        output = F.log_softmax(x, dim=1)

        return torch.squeeze(output)
