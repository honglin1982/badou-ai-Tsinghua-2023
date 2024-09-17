# -*- coding: utf-8 -*-
import time

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader


class Cifar10Module(nn.Module):
    def __init__(self):
        super(Cifar10Module, self).__init__()
        self.conv1 = nn.Sequential(  # 输入 1 * 24 * 24
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1)  # 输出 64 * 8 * 8
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1)  # 输出 64 * 4 * 4
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 6 * 6, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=192),
            nn.ReLU(),
            nn.Linear(in_features=192, out_features=10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class Model:

    def __init__(self, net, device, lr=0.001):
        self.net = net
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func.to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.total_train_step = 0

    def train(self, loader, epochs=3):
        for epoch in range(epochs):
            for i, data in enumerate(loader):
                start_time = time.time()
                train_data, train_labels = data
                train_data = train_data.to(device)
                train_labels = train_labels.to(device)

                result = self.net(train_data)
                loss = self.loss_func(result, train_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.total_train_step += 1
                if self.total_train_step % 100 == 0:
                    print('epoch: {}, 耗时：{}'.format(epoch, time.time() - start_time))
                    print("训练次数: {}, Loss: {}".format(self.total_train_step, loss))
        print("训练完成")

    def evaluate(self, loader):
        right_num = 0
        total = 0
        # 不保存梯度
        with torch.no_grad():
            for data in loader:
                test_data, test_labels = data
                test_data = test_data.to(device)
                test_labels = test_labels.to(device)
                # 计算
                outputs = self.net(test_data)
                # 获取每个最大值的索引
                predicted = torch.argmax(outputs, 1)
                total += len(test_labels)
                # 取出正确数量
                right_num += (predicted == test_labels).sum().item()
        print('正确率：', right_num/total)


if __name__ == '__main__':
    batch_size = 100
    # 使用compose串联多个变换操作
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(24),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=(0, 0.8), contrast=(0.2, 1.8)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5),
        torchvision.transforms.Resize(24)
    ])
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])

    train_data = torchvision.datasets.CIFAR10('./cifar-10-pytorch', train=True, transform=transform_train,
                                              download=True)
    test_data = torchvision.datasets.CIFAR10('./cifar-10-pytorch', train=False, transform=transform_test, download=True)
    train_loader = DataLoader(train_data, batch_size, True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(train_data, batch_size, True, pin_memory=True)

    device = torch.device('cuda')
    net = Cifar10Module()
    net.to(device)
    model = Model(net, device, 0.001)
    model.train(train_loader, epochs=100)
    model.evaluate(test_loader)
