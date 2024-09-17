# -*- coding: utf-8 -*-
# File  : train.py
# Author: HeLei
# Date  : 2023/8/8

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
import torchvision
import time
from model import AlexNet
import untils

# 训练

# 指定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# 初始学习率
lr = 0.01

# 初始化模型
model = AlexNet.AlexNet().to(device=device)

# 构造损失函数和优化器
loss_function = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)

# 动态更新学习率
schedule = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.6, last_epoch=-1)

# 加载数据集
train_loader, test_loader = untils.data_loader()


def train(epochs):
    loss_list = []  # 存放loss
    bestacc = 0.0  # 最好的准确率
    bestepoch = 0  # 最好的训练轮次
    epochl = []
    accl = []
    start = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        account = 0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device=device), labels.to(device=device)

            # 将数据送入模型训练
            output = model(inputs)  # 预测值

            # 计算损失
            loss = loss_function(output, labels).to(device=device)

            # 重置梯度
            opt.zero_grad()

            # 反向传播
            loss.backward()

            # 根据反向传播的梯度值优化更新参数
            opt.step()

            running_loss += loss.item()
            loss_list.append(loss.item())

            account += inputs.size(0)

            # 每一百个 batch 查看一下 loss
            # if (i + 1) % 100 == 0:
            #     print(f"epoch = {epoch + 1},batch = {i + 1},loss = {running_loss / 100:.6f}")
            #     running_loss = 0.0

        # # 每一轮结束输出一下当前的学习率 lr
        # lr_1 = opt.param_groups[0]['lr']
        # print("learn_rate:%.15f" % lr_1)
        # schedule.step()

        acc = test()
        if acc > bestacc:
            bestacc, bestepoch = acc, epoch + 1
            torch.save(model, './model/cifar10_model_{}.pt'.format(epoch + 1))
        accl.append(acc)
        epochl.append(epoch + 1)
        loss_list.append(running_loss)
        print('Epoch{}:\n\tloss:{:.3f} acc:{:.3f} lr:{} '
              'bestepoch:{} bestacc:{:.3f} time:{:.3f}'.format(
            epoch + 1, running_loss / account, acc, schedule.get_last_lr(), bestepoch, bestacc, time.time() - start
        ))

    # 可视化训练过程
    plt.plot(epochl, loss_list)
    plt.plot(epochl, accl)
    plt.legend(['loss', 'acc'])
    plt.savefig('./loss_and_acc.png')


def test():
    model.eval()
    correct = 0.0
    total = 0
    # 测试模式不需要反向传播更新梯度
    with torch.no_grad():
        # print("=========================test=========================")
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # 预测值

            pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
            total += inputs.size(0)
            correct += torch.eq(pred, labels).sum().item()

    # print("Accuracy of the network on the 10000 test images:%.2f %%" % (100 * correct / total))
    # print("======================================================")

    return 100 * correct / total

if __name__ == '__main__':
    epochs = 10
    train(epochs)