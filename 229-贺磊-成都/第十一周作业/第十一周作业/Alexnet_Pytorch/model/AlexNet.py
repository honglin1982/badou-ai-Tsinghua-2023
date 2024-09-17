# -*- coding: utf-8 -*-
# File  : AlexNet.py
# Author: HeLei
# Date  : 2023/8/8
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_class=10):
        super(AlexNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False),  # 55*55*96
            nn.ReLU(True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # 27*27*96

            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(384),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=num_class),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        print()
        x = x.view(x.size(0), 256 * 6 * 6)
        # x.size(0)表示x的第一个维度的大小，即batch_size。
        # x.view(x.size(0), 256 * 6 * 6)表示将x的形状重塑为(batch_size, 256 * 6 * 6)。
        # 其中，x.view()是PyTorch中的一个函数，用于重塑张量的形状。
        # x.size(0)表示新张量的第一个维度的大小，即batch_size。
        # 256 * 6 * 6表示新张量的总元素个数，即将原张量展平为一维。
        x = self.classifier(x)
        return x


# if __name__ == '__main__':
#     model = AlexNet()
#     print(model)
#     input = torch.randn(8, 3, 224, 224)
#     out = model(input)
#     print(out.shape)
