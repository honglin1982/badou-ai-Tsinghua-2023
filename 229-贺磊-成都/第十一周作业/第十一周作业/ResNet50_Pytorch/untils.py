# -*- coding: utf-8 -*-
# File  : untils.py
# Author: HeLei
# Date  : 2023/8/1

from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def data_loader(batch_size):
    """
    数据加载
    :param batch_size: 批次大小
    :return:
    """
    # 训练集
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # R,G,B三通道归一化的的均值和方差
    ]
    )
    # 测试集
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = datasets.CIFAR10(root='./data', download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = datasets.CIFAR10(root='./data', download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = data_loader(16)
    print(train_loader)
    print(len(test_loader))
