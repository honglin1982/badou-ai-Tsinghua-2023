# _*_ coding : utf-8 _*_
# @Time : 2023/8/16 9:25
# @Author : weixing
# @FileName : DarkNet53
# @Project : cv

import torch.nn as nn
import torch


# 残差模块  下采样卷积+残差块
class ResidualBlock(nn.Module):
    def __init__(self, input_channel, mid_channel):
        super(ResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(0.1),

            nn.Conv2d(mid_channel, input_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_channel),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        residual = x

        out = self.residual(x)

        out = residual + out

        return out


'''
下采样卷积+残差块
input_channel 下采样卷积输入通道，输出通道为input_channel*2
layer 残差块数目
'''


def _make_layer(input_channel, layer):
    layers = [nn.Conv2d(input_channel, input_channel * 2, kernel_size=3, stride=2, padding=1)]

    for _ in range(layer):
        residual = ResidualBlock(input_channel * 2, input_channel)
        layers.append(residual)

    return nn.Sequential(*layers)


'''
darknet53基本结构：
卷积+(下采样卷积+1残差块)+(下采样卷积+2残差块)+(下采样卷积+8残差块)+(下采样卷积+8残差块)+(下采样卷积+4*残差块)
'''


class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()

        self.layer_0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )  # input [batch_size,416,416,3] out [batch_size,416,416,32]

        self.layer_1 = _make_layer(32, 1)  # input [batch_size,416,416,32] out [batch_size,208,208,64]

        self.layer_2 = _make_layer(64, 2)  # input [batch_size,208,208,64] out [batch_size,104,104,128]

        self.layer_3 = _make_layer(128, 8)  # input [batch_size,104,104,128] out [batch_size,52,52,256]

        self.layer_4 = _make_layer(256, 8)  # input [batch_size,52,52,256] out [batch_size,26,26,512]

        self.layer_5 = _make_layer(512, 4)  # input [batch_size,26,26,512] out [batch_size,13,13,1024]

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        out_52 = self.layer_3(x)
        out_26 = self.layer_4(out_52)
        out_13 = self.layer_5(out_26)

        return out_52, out_26, out_13


'''
pretrained 是否使用预训练模型
'''

Model_File_Url = "./model/model_13.pth"


def darknet53(pretrained=False, **kwargs):
    model = DarkNet53()
    if pretrained:
        model.load_state_dict(torch.load(Model_File_Url))
    return model

# model = DarkNet53()
# print(model)
