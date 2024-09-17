# _*_ coding : utf-8 _*_
# @Time : 2023/7/18 17:16
# @Author : weixing
# @FileName : pytorch-minist
# @Project : cv


'''
准备数据集：使用 MNIST 数据集作为手写数字数据集。该数据集包含 60000 张 28x28 的灰度图像作为训练数据和 10000 张图像作为测试数据。
    将数据集分为训练集和测试集，比例为 6:1。
定义模型：使用 PyTorch 的 nn.Module 类定义一个神经网络模型。
编译模型：使用 PyTorch 的 nn.CrossEntropyLoss() 函数定义损失函数，使用 optimizer 类定义优化器，如 SGD 或 Adam。
        将模型、损失函数和优化器组合在一起，使用 model.train() 和 model.eval() 方法来训练和评估模型。
测试模型：使用测试集对模型进行评估。在每个 epoch 后，使用测试集计算模型的准确率和损失函数值，并记录最佳性能指标（如准确率或损失函数值）。
预测结果：使用训练好的模型对新的手写数字图像进行预测。将图像转换为合适的张量格式，输入模型中得到预测结果，并将预测结果转换为数字输出。
'''

import torch
import torch.nn as nn
from torchvision import datasets, transforms


# 准备数据集
def minist_loadData():
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    print("训练数据集数量：", len(train_dataset))
    print("测试数据集数量：", len(test_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    return train_loader, test_loader

"""
比较两个向量中有多少数字相等
"""
def count_equal_elements(vec1, vec2):
    count = 0
    for i in range(len(vec1)):
        if vec1[i] == vec2[i]:
            count += 1
    return count

class NetWork(nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)  #log_softmax下降效果显著，但是速度很慢；softmax下降效果不理想

class Model():
    def __init__(self, network):
        self.network = network
        # 定义损失函数、优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01)

    def train(self, epoch, train_loader):
        print("start training")
        # 训练模型
        for epoch in range(epoch):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self.network(inputs)  # inputs [64, 1, 28, 28]  labels [64, 1]  outputs #[64, 10]
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print('train;[Epoch %d] Loss: %.4f' % (epoch + 1, running_loss / len(train_loader)))

        print("train end")
    def eval(self, test_loader):
        print("start test")
        # 测试模型
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.network(inputs)  # [64, 10]
                _, predicted = torch.max(outputs.data, 1)  # [64]
                correct += count_equal_elements(labels, predicted)
                total += labels.size(0)
            print("predict;测试集总个数 %d, 预测正确的个数 %d, 准确率 %.4f" % (total, correct, correct / total))

            # 保存参数
            if correct/total > 0.97:
                print("model to be saved")
                torch.save(self.network.state_dict(), "./model/torch-mnist.h5")

        print("test end")
def main():
    # 加载数据集
    train_loader, test_loader = minist_loadData()
    # 网络
    network = NetWork()
    # 模型
    model = Model(network)
    # 训练
    epoch = 10
    model.train(epoch, train_loader)
    # 测试
    model.eval(test_loader)

if __name__=='__main__':
    main()