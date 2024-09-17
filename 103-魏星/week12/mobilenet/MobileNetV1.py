# _*_ coding : utf-8 _*_
# @Time : 2023/8/2 15:07
# @Author : weixing
# @FileName : MobileNet
# @Project : cv

import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBnRelu_normal(input_channel, out_channel, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(input_channel, out_channel, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


def ConvBnRelu_depthWise(input_channel, out_channel, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(input_channel, input_channel, kernel_size, stride, padding, groups=input_channel),
        nn.BatchNorm2d(input_channel),
        nn.ReLU(inplace=True)
    )


class DepthSeparableConv(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(DepthSeparableConv, self).__init__()

        self.depth_wise_conv = ConvBnRelu_depthWise(input_channel, input_channel, kernel_size=kernel_size,
                                                    stride=stride, padding=padding)

        self.normal_conv = ConvBnRelu_normal(input_channel, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depth_wise_conv(x)

        x = self.normal_conv(x)

        return x


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        self.conv_1 = ConvBnRelu_normal(3, 32, kernel_size=3, stride=2, padding=1)
        # input[N,3,224,224] out[N,32,112,112]
        self.depthConv_1 = DepthSeparableConv(32, 64, kernel_size=3, padding=1)  # input[N,32,112,112] out[N,64,112,112]
        self.depthConv_2 = DepthSeparableConv(64, 128, kernel_size=3, stride=2, padding=1)
        # input[N,64,112,112] out[N,128,56,56]
        self.depthConv_3 = DepthSeparableConv(128, 128, kernel_size=3, padding=1)  # input[N,128,56,56] out[N,128,56,56]
        self.depthConv_4 = DepthSeparableConv(128, 256, kernel_size=3, stride=2, padding=1)
        # input[N,128,56,56] out[N,256,28,28]
        self.depthConv_5 = DepthSeparableConv(256, 256, kernel_size=3, padding=1)
        # input[N,256,28,28] out[N,256,28,28]
        self.depthConv_6 = DepthSeparableConv(256, 512, kernel_size=3, stride=2, padding=1)
        # input[N,256,28,28] out[N,512,14,14]

        self.depthConv_7x5 = nn.Sequential(
            DepthSeparableConv(512, 512, kernel_size=3, stride=1, padding=1),
            DepthSeparableConv(512, 512, kernel_size=3, stride=1, padding=1),
            DepthSeparableConv(512, 512, kernel_size=3, stride=1, padding=1),
            DepthSeparableConv(512, 512, kernel_size=3, stride=1, padding=1),
            DepthSeparableConv(512, 512, kernel_size=3, stride=1, padding=1)
        )
        # input[N,512,14,14] out[N,512,14,14]

        self.depthConv_12 = DepthSeparableConv(512, 1024, kernel_size=3, stride=2, padding=1)
        # input[N,512,14,14] out[N,1024,7,7]

        self.depthConv_13 = DepthSeparableConv(1024, 1024, kernel_size=3, stride=1, padding=1)
        # input[N,1024,7,7] out[N,1024,7,7]

        self.avgPool = nn.AvgPool2d(kernel_size=7)  # input[N,1024,7,7] out[N,1024,1,1]

        self.fc = nn.Linear(1024, num_classes)  # input[N,1024] out[N,num_classes]

    def forward(self, x):
        x = self.conv_1(x)
        x = self.depthConv_1(x)
        x = self.depthConv_2(x)
        x = self.depthConv_3(x)
        x = self.depthConv_4(x)
        x = self.depthConv_5(x)
        x = self.depthConv_6(x)
        x = self.depthConv_7x5(x)
        x = self.depthConv_12(x)
        x = self.depthConv_13(x)

        x = self.avgPool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)

        return F.log_softmax(x, dim=1)


# model = MobileNetV1(1000)
# print(model)

