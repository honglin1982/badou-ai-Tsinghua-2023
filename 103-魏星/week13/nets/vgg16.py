# _*_ coding : utf-8 _*_
# @Time : 2023/8/4 16:46
# @Author : weixing
# @FileName : vgg16
# @Project : cv

import torch
import torch.nn as nn

channels = [64, 64, 'max_pool', 128, 128, 'max_pool', 256, 256, 256, 'max_pool', 512, 512, 512, 'max_pool', 512, 512,
            512]
Model_File_Url = "./model/model_13.pth"


# 构造backbone网络
def create_layers(out_channels=None, batch_norm=False):
    if out_channels is None:
        out_channels = channels
    input_channel = 3
    layers = []
    for out_channel in out_channels:
        if out_channel == 'max_pool':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(input_channel, out_channel, kernel_size=3, stride=1, padding=1))
            input_channel = out_channel
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channel))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)
    # features = nn.Sequential(
    #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # input [M,N,3] output [M,N,64]
    #     nn.ReLU(inplace=False),
    #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # input [M,N,64] output [M,N,64]
    #     nn.ReLU(inplace=False),
    #
    #     nn.MaxPool2d(kernel_size=2, stride=2),  # input [M,N,64] output [M/2,N/2,64]
    #
    #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # input [M/2,N/2,64] output [M/2,N/2,128]
    #     nn.ReLU(inplace=False),
    #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # input [M/2,N/2,128] output [M/2,N/2,128]
    #     nn.ReLU(inplace=False),
    #
    #     nn.MaxPool2d(kernel_size=2, stride=2),  # input [M/2,N/2,128] output [M/4,N/4,128]
    #
    #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # input [M/4,N/4,128] output [M/4,N/4,256]
    #     nn.ReLU(inplace=False),
    #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # input [M/4,N/4,256] output [M/4,N/4,256]
    #     nn.ReLU(inplace=False),
    #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # input [M/4,N/4,256] output [M/4,N/4,256]
    #     nn.ReLU(inplace=False),
    #
    #     nn.MaxPool2d(kernel_size=2, stride=2),  # input [M/4,N/4,256] output [M/8,N/8,256]
    #
    #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # input [M/8,N/8,256] output [M/8,N/8,512]
    #     nn.ReLU(inplace=False),
    #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # input [M/8,N/8,512] output [M/8,N/8,512]
    #     nn.ReLU(inplace=False),
    #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # input [M/8,N/8,512] output [M/8,N/8,512]
    #     nn.ReLU(inplace=False),
    #
    #     nn.MaxPool2d(kernel_size=2, stride=2),  # input [M/8,N/8,512] output [M/16,N/16,512]
    #
    #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # input [M/16,N/16,512] output [M/16,N/16,512]
    #     nn.ReLU(inplace=False),
    #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # input [M/16,N/16,512] output [M/16,N/16,512]
    #     nn.ReLU(inplace=False),
    #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # input [M/16,N/16,512] output [M/16,N/16,512]
    #     nn.ReLU(inplace=False)
    # )
    #
    # return features


class VGG16(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG16, self).__init__()

        self.features = features

        self.avgPool = nn.AdaptiveAvgPool2d((7, 7))  # input [M/16,N/16,512] output [7,7,512]

        # 分类,用卷积代理全连接
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0),  # input [7,7,512] output [1,1,4096]
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Conv2d(4096, 4096, kernel_size=3, stride=1, padding=1),  # input [1,1,4096] output [1,1,4096]
            nn.ReLU(inplace=True),

            nn.Conv2d(4096, num_classes, kernel_size=3, stride=1, padding=1)
            # input [1,1,4096] output [1,1,num_classes]
        )

    def forward(self, x):
        x = self.features(x)

        x = self.avgPool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x


# 获取vgg16 特征层、分类层
def get_vgg16(pretrained=False):
    model = VGG16(create_layers(out_channels=channels, batch_norm=False))
    if pretrained:
        model.load_state_dict(torch.load(Model_File_Url))
    # print(model)
    # 获取特征提取，out [M/16,N/16,1024]
    features = list(model.features)[:30]
    # 获取分类部分，除去Dropout部分
    classifier = list(model.classifier)
    del classifier[3]
    del classifier[0]

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier
