# -*- coding: utf-8 -*-
# File  : train.py
# Author: HeLei
# Date  : 2023/8/1

import torch
import torchsummary
from torch import nn, optim
from nets import ResNet50
import argparse
from torch.utils.tensorboard import SummaryWriter
import untils
import time

# 1、设置GPU训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device!")


# 加载模型
model = ResNet50.ResNet50().to(device)
torchsummary.summary(model,(3,224,224)) # 网络结构


# 模型训练
epochs = 100
Batch_Size = 64  # 批处理尺寸
lr = 0.001  # 初始学习率
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数，多分类问题
optimizer = optim.SGD(model.parameters(), lr=lr,
                      momentum=0.8, weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减
# 动态更新学习率
schedule = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6, last_epoch=-1)

train_loss = []
train_acc = []
validate_loss = []
validate_acc = []
epoch_best_acc = []

# 添加tensorboard画图可视化
writer = SummaryWriter("logs/logs_train")

# 加载数据集
train_loader,test_loader = untils.data_loader(Batch_Size)

def train(model,loss_fn,optimizer):

    size = len(train_loader)

    train_loss,train_acc = 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device=device), labels.to(device=device)

        # 将数据送入模型训练
        output = model(inputs)  # 预测值

        # 计算损失
        loss = loss_fn(output, labels).to(device=device)

        # 重置梯度
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 根据反向传播的梯度值优化更新参数
        optimizer.step()

        # 记录acc与loss
        train_acc += (output.argmax(1) == labels).type(torch.float).sum().item()
        train_loss += loss.item()



def test():
    # 记录测试的次数
    total_test_step = 0
    model.eval()
    correct = 0.0
    total = 0
    total_test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss = loss_fn(outputs, labels)
            total_test_loss += test_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total

    print(f"在测试集上的loss:{total_test_loss}")
    print(f"在测试集的准确率：{acc}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1
    return acc


if __name__ == '__main__':
    train()
