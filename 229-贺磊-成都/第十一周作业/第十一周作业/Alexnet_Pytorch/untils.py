# -*- coding: utf-8 -*-
# File  : untils.py
# Author: HeLei
# Date  : 2023/7/25

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision


def data_loader():
    batch_size = 16
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(224),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.RandomHorizontalFlip(p=0.5)]
    )
    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
