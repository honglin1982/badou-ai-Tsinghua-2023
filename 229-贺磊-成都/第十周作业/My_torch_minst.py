# -*- coding: utf-8 -*-
# File  : My_torch_minst.py
# Author: HeLei
# Date  : 2023/7/14

"""
pytorch中自带的数据集有两个上层的API提供，分别是torchvision和torchtext
其中：

1.torchvision提供了对图片数据处理相关的API和数据

 数据位置：torchvision.datasets,例如：torchvision.datasets.MNIST(手写数字图片数据)

2.torchtext提供了对文本数据处理相关的API和数据

 数据位置：torchtext.datasets,例如：torchtext.datasets.IMDB(电影评论文本数据)

"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms


def load_data():
    # 预处理数据的transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0, ], [0.5, ])
    ]
    )
    # 导入数据集
    train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # 定义数据集加载器
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=2)

    return train_loader, test_loader


class My_torch_net:
    def __init__(self, net, cost_function, optimist):
        self.net = net
        self.cost_function = self.create_cost(cost_function)
        self.optimizer = self.create_optimizer(optimist)

    # 构造损失函数
    @staticmethod
    def create_cost(self, cost_function):
        support_cost = {
            "CROSS_ENTROPY": nn.CrossEntropyLoss(),  # 交叉熵损失函数
            "MSE": nn.MSELoss()  # 均方差损失函数
        }
        return support_cost[cost_function]

    # 构造优化器
    @staticmethod
    def create_optimizer(self, optimizer, **kwargs):
        support_optim = {
            "SGD": optim.SGD(self.net.parameters(), lr=0.1, **kwargs),
            "Adam": optim.Adam(self.net.parameters(), lr=0.01, **kwargs),
            "RMSP": optim.RMSprop(self.net.parameters(), lr=0.001, **kwargs)
        }
        return support_optim[optimizer]

    # 训练
    @staticmethod
    def train(self, train_loader, epochs=5):
        print("Training...")
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                # 前向传播 + 反向传播 + 优化器
                outputs = self.net(inputs)
                loss = self.cost_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print("Finished Training...")

    @staticmethod
    def evaluate(self, test_loader):
        print("Evaluating...")
        correct = 0
        total = 0
        # 测试和推理时不需要计算梯度
        with torch.no_grad():
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)

                correct += (predicted == predicted).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

if __name__ == '__main__':
    net = MnistNet()
    model = My_torch_net(net,'CROSS_ENTROPY','RMSP')
    train_loader, test_loader = load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
