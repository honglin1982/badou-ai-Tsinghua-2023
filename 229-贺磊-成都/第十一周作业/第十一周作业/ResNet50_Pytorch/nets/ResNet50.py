# -*- coding: utf-8 -*-
# File  : ResNet50.py
# Author: HeLei
# Date  : 2023/7/31

import torch
from torch import nn
from torchsummary import torchsummary


# IdentityBlock层
class IdentityBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, filters1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(filters1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filters1, filters2, kernel_size, stride=1, padding=autopad(kernel_size), bias=False),
            nn.BatchNorm2d(filters2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters2, filters3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(filters3)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x = x1 + x
        self.relu(x)
        return x


# ConvBlock层
class ConvBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters, stride=2):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, filters1, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(filters1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filters1, filters2, kernel_size, stride=1, padding=autopad(kernel_size), bias=False),
            nn.BatchNorm2d(filters2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters2, filters3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(filters3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel, filters3, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(filters3)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x2 = self.conv4(x)
        x = x1 + x2
        self.relu(x)
        return x


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False,
                      padding_mode="zeros"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(
            ConvBlock(64, 3, [64, 64, 256], stride=1),
            IdentityBlock(256, 3, [64, 64, 256]),
            IdentityBlock(256, 3, [64, 64, 256])
        )
        self.conv3 = nn.Sequential(
            ConvBlock(256, 3, [128, 125, 512]),
            IdentityBlock(512, 3, [128, 128, 512]),
            IdentityBlock(512, 3, [128, 128, 512]),
            IdentityBlock(512, 3, [128, 128, 512])
        )
        self.conv4 = nn.Sequential(
            ConvBlock(512, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024])
        )
        self.conv5 = nn.Sequential(
            ConvBlock(1024, 3, [512, 512, 2048]),
            IdentityBlock(2048, 3, [512, 512, 2048]),
            IdentityBlock(2048, 3, [512, 512, 2048])
        )
        # self.pool = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def autopad(kernel, padding=None):
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
    return padding


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet50().to(device)
    torchsummary.summary(model, (3, 224, 224))
