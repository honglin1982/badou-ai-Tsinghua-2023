# _*_ coding : utf-8 _*_

import os
import json
import sys

import torch
import torch.nn as nn
import torchvision.datasets
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_mobilenet import MobileNet

batch_size = 64  # 批次
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = 0  # 给Dataloader设置worker数量


class MobilenetModel:
    """ cifar10 训练模型 GPU/CPU"""
    epoch_num = 3  # 训练轮次
    lr = 0.001  # 学习率
    # best_acc = 0.87  # 最低准确率，超过这个值则保存模型
    model_save_path = './torch-cifar10-mobilenet-v1.pth'

    def __init__(self, net: MobileNet, cost_str=None, optimist_str=None):
        self.net = net
        self.criterion = self.create_cost(cost_str)
        self.optimizer = self.create_optimizer(optimist_str, lr=self.lr, momentum=0.9)

    def create_cost(self, cost_str):
        # 损失函数
        support_cost = {
            "CROSS_ENTROPY": nn.CrossEntropyLoss(),
            "MSE": nn.MSELoss(),
        }
        return support_cost[cost_str]

    def create_optimizer(self, optimist_str, **rests):
        """
        momentum是SGD优化器的一个超参数，它的作用是在每次迭代时，将上一次迭代的梯度乘以一个惯性权重，加入到当前的梯度中，
        从而加快收敛速度。具体来说，momentum越大，惯性越大，收敛速度越快；momentum越小，惯性越小，收敛速度越慢。
        """
        # 优化器包
        support_optimist = {
            "SGD": optim.SGD(self.net.parameters(), **rests),
            "ADAM": optim.Adam(self.net.parameters(),  **rests),
            "RMSP": optim.RMSprop(self.net.parameters(), **rests),
        }
        return support_optimist[optimist_str]

    def train(self, train_loader: DataLoader):
        """
        1.先将梯度归零：
            一次正向传播得到预测值
            计算损失值
        2.反向传播得到每个参数的梯度值
        3.根据梯度进行参数更新
        """
        for epoch in range(self.epoch_num):
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for i, data in enumerate(train_bar, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # 1.先将梯度归零: 目前主流的深度学习模型的优化器都是随机批次梯度下降法，即对一个batchsize数据去求平均梯度，
                # 根据得到的这个平均梯度去更新所有参数。因此，每个batchsize数据的梯度是唯一的，每次重新开始一个批次的计算必须先将参数之前的对应梯度清零。
                self.optimizer.zero_grad()
                # 一次正向传播得到预测值, 输出的是 32行数据，每行10列的概率值
                outputs = self.net(inputs)
                # 计算损失值
                loss = self.criterion(outputs, labels)
                # 2.反向传播得到每个参数的梯度值
                loss.backward()
                # 3.根据梯度进行参数更新
                self.optimizer.step()
                running_loss += loss.item()  # .item() 在这里是提取数据
                if i % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{self.epoch_num}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}, avgloss: {round(running_loss / 100, 6)}')
                train_bar.desc = f'Epoch [{epoch + 1}/{self.epoch_num}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}'

        print('Finished Training ...')

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict : 不用计算梯度情况
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self.net(images)  # torch.Size([32, 10])
                # predicted = torch.argmax(outputs, dim=1)
                _v, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                correct += torch.eq(predicted, labels).sum().item()
            print('Accuracy of the network on the test images: %d %%' % (correct / total * 100))
            # # 保存参数
            # if correct / total > best_acc:
            #     print("model to be saved")
            #     torch.save(self.network.state_dict(), Model_File_Url)
        torch.save(self.net.state_dict(), self.model_save_path)
        print('Finished Testing ...')


def mnist_load_data():
    data_transformer = {
        "train": transforms.Compose([
            # 这里的scale指的是面积，ratio是宽高比
            # 具体实现每次先随机确定scale和ratio，可以生成w和h，然后随机确定裁剪位置进行crop
            # 最后是resize到target size
            # 随机缩放裁剪
            transforms.RandomResizedCrop(224),
            # transforms.RandomRotation(degrees=15),  # 随机旋转
            # 水平翻转
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
            # 具体来说，mean和std的取值范围为[0,1]，即将图像的每个通道的像素值都除以255后再减去均值mean，再除以标准差std。这样做可以使得数据分布更加均匀，并且有利于模型的训练和收敛。
            # 例如，上述代码中的mean和std分别为[0.5, 0.5, 0.5]和[0.5, 0.5, 0.5]，表示将每个通道的归一化后的像素值都减去0.5后，再除以0.5。
            # 这样做可以使输入数据的均值为0，方差为1，有助于提高模型的稳定性和收敛速度。
        ]),
        # 测试阶段是执行缩放和中心裁剪
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    }
    # 加载CIFAR-10数据集
    # 下载
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                 transform=data_transformer["train"])
    train_num = len(train_dataset)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                transform=data_transformer["test"])
    test_num = len(test_dataset)
    # load
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("using {} images for training, {} images for test.".format(train_num, test_num))
    return train_loader, test_loader


def main():
    # 加载数据集
    train_loader, test_loader = mnist_load_data()
    network = MobileNet(num_classes=10)
    network.to(device=device)  # 放到GPU/CPU上
    model = MobilenetModel(net=network, cost_str='CROSS_ENTROPY', optimist_str='SGD')
    # 训练
    model.train(train_loader=train_loader)
    # 测试
    model.evaluate(test_loader=test_loader)


if __name__ == '__main__':
    main()



