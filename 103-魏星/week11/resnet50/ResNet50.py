# _*_ coding : utf-8 _*_
# @Time : 2023/7/24 17:48
# @Author : weixing
# @FileName : netWork
# @Project : cv

import torch
import torch.nn as nn


# 卷积+BN
class BasicNet(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0):
        super(BasicNet, self).__init__()
        self.basicConv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x = self.basicConv(x)
        return x


# conv_block 或 identity_block
class BlockNet(nn.Module):
    def __init__(self, input_channel, middle_channel, output_channel, stride=1):
        super(BlockNet, self).__init__()

        # 不相等则需要处理shortcut(利用1*1卷积改变通道数)
        self.isShortCutDown = input_channel == output_channel

        self.blockNet = nn.Sequential(
            BasicNet(input_channel, middle_channel, kernel_size=1),
            BasicNet(middle_channel, middle_channel, kernel_size=3, stride=stride, padding=1),
            BasicNet(middle_channel, output_channel, kernel_size=1)
        )

        # 残差处理shortcut
        if input_channel != output_channel:
            self.shortcut = BasicNet(input_channel, output_channel, kernel_size=1, stride=stride)

    def forward(self, x):
        result = self.blockNet(x)

        if not self.isShortCutDown:
            shortcut = self.shortcut(x)
            result = result + shortcut
        else:
            result = result + x

        return result


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()

        self.conv1_x = BasicNet(3, 64, kernel_size=7, stride=2, padding=3)  # input [224,224,3] output [112,112,64]
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # input [112,112,64] output[56,56,64]

        self.conv2_x = nn.Sequential(
            BlockNet(64, 64, 256),  # conv_block  #input[56,56,64]->[56,56,64]->[56,56,64]->[56,56,256]
            BlockNet(256, 64, 256),  # identity_block  #input[56,56,256]->[56,56,64]->[56,56,64]->[56,56,256]
            BlockNet(256, 64, 256)  # identity_block #input[56,56,256]->[56,56,64]->[56,56,64]->[56,56,256]
        )

        self.conv3_x = nn.Sequential(
            BlockNet(256, 128, 512, stride=2),  # conv_block input[56,56,256]->[56,56,128]->[28,28,128]->[28,28,512]
            BlockNet(512, 128, 512),  # identity_block  input[28,28,512]->[28,28,128]->[28,28,128]->[28,28,512]
            BlockNet(512, 128, 512),  # identity_block  input[28,28,512]->[28,28,128]->[28,28,128]->[28,28,512]
            BlockNet(512, 128, 512)  # identity_block  input[28,28,512]->[28,28,128]->[28,28,128]->[28,28,512]
        )

        self.conv4_x = nn.Sequential(
            BlockNet(512, 256, 1024, stride=2),  # conv_block input[28,28,512]->[28,28,256]->[14,14,256]->[14,14,1024]
            BlockNet(1024, 256, 1024),  # identity_block input[14,14,1024]->[14,14,256]->[14,14,256]->[14,14,1024]
            BlockNet(1024, 256, 1024),  # identity_block input[14,14,1024]->[14,14,256]->[14,14,256]->[14,14,1024]
            BlockNet(1024, 256, 1024),  # identity_block input[14,14,1024]->[14,14,256]->[14,14,256]->[14,14,1024]
            BlockNet(1024, 256, 1024),  # identity_block input[14,14,1024]->[14,14,256]->[14,14,256]->[14,14,1024]
            BlockNet(1024, 256, 1024)  # identity_block input[14,14,1024]->[14,14,256]->[14,14,256]->[14,14,1024]
        )

        self.conv5_x = nn.Sequential(
            BlockNet(1024, 512, 2048, stride=2),  # conv_block input[14,14,1024]->[14,14,512]->[7,7,512]->[7,7,2048]
            BlockNet(2048, 512, 2048),  # identity_block input[7,7,2048]->[7,7,512]->[7,7,512]->[7,7,2048]
            BlockNet(2048, 512, 2048)  # identity_block input[7,7,2048]->[7,7,512]->[7,7,512]->[7,7,2048]
        )

        self.averagePool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # input[7,7,2048]-> [1,1,2048]

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1_x(x)
        x = self.maxPool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.averagePool(x)

        x = torch.squeeze(x)

        x = self.fc(x)

        return x

