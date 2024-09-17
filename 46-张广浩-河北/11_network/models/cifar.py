# -*- coding: utf-8 -*-
# @Time    : 2023/7/21 21:25
# @Author  : zgh
# @FileName: cifar.py
# @Software: PyCharm
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

class CIFAMODULE(nn.Module):
    def __init__(self):
        super(CIFAMODULE, self).__init__()
        self.model1=Sequential(
            Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)

        )

    def forward(self,x):
        x = self.model1(x)
        return x

if __name__ == '__main__':
    cifa = CIFAMODULE()
