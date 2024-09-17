# -*- coding: utf-8 -*-
# @Time    : 2023/7/21 19:47
# @Author  : zgh
# @FileName: Alexnet.py
# @Software: PyCharm

import torch.optim
import torchvision.datasets
# 准备训练数据集
from torch import nn


# 227*227
class Alexnet(nn.Module):
    def __init__(self,num_classes=10):
        super(Alexnet, self).__init__()
        self.features = nn.Sequential(
            # 输入通道数为3，因为图片为彩色，三通道
            # 而输出96、卷积核为11*11，步长为4，是由AlexNet模型决定的，后面的都同理
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(in_features=6 * 6 * 256, out_features=4096),
            nn.ReLU(),
            # AlexNet采取了DropOut进行正则，防止过拟合
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # 最后一层，输出1000个类别，也是我们所说的softmax层
            nn.Linear(4096, 5)
        )



    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x

if __name__ == '__main__':
    net = Alexnet(num_classes=10)
    print(net)






