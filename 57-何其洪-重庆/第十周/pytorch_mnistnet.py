# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        # 定义三层全链接神经网络
        # in_features 输入张量形状，out_features 输出张量形状
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        # 先调用神经元在调用激活函数
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.softmax(self.fc3(x), dim=1)
        return x


class Model:
    def __init__(self, net):
        self.net = net
        self.cost = nn.CrossEntropyLoss()
        self.optimizer = optim.RMSprop(self.net.parameters(), 0.001)

    def train(self, loader, epochs=3):
        for epoch in range(epochs):
            for i, data in enumerate(loader):
                train_data, train_labels = data
                # 梯度清零
                self.optimizer.zero_grad()
                # 调用神经网络
                outputs = self.net(train_data)
                # 计算损失
                loss = self.cost(outputs, train_labels)
                # 反向传播
                loss.backward()
                self.optimizer.step()
        print('训练完成')

    def evaluate(self, loader):
        right_num = 0
        total = 0
        # 不保存梯度
        with torch.no_grad():
            for data in loader:
                test_data, test_labels = data
                # 计算
                outputs = self.net(test_data)
                # 获取每个最大值的索引
                predicted = torch.argmax(outputs, 1)
                total += len(test_labels)
                # 取出正确数量
                right_num += (predicted == test_labels).sum().item()
        print('正确率：', right_num/total)


if __name__ == '__main__':
    # 加载数据集
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0, ], [1, ])
    ])
    train_loader = DataLoader(torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True),
                              batch_size=100, shuffle=True)
    test_loader = DataLoader(torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True),
                             batch_size=100, shuffle=True)

    # 创建网络
    net = MnistNet()
    # 创建模型
    model = Model(net)
    # 训练模型
    model.train(train_loader, 3)
    # 验证模型
    model.evaluate(test_loader)

