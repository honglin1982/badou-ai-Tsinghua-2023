# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import os
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datasets.Mydataset import My_Dataset, My_Dataset_test
from models.Alexnet import Alexnet
from models.VGG import Vgg16

def train():
    batch_size = 32     # 批量训练大小
    model = Vgg16() # 创建模型
    print(model)
    # 将模型放入GPU中
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(params=model.parameters(),lr=0.0002)
    # 加载数据
    train_path = r"D:\python\data\flowers\net_train_images"
    train_set = My_Dataset(train_path,transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    # 训练20次
    epoch = 20
    for i in range(epoch):
        loss_temp = 0  # 临时变量
        for j,(batch_data,batch_label) in enumerate(train_loader):
            # 数据放入GPU中
            batch_data,batch_label = batch_data.cuda(),batch_label.cuda()
            # 梯度清零
            optimizer.zero_grad()
            # 模型训练
            prediction = model(batch_data)
            # 损失值
            loss = loss_func(prediction,batch_label)
            loss_temp += loss.item()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 每25个批次打印一次损失值
        print('[%d] loss: %.3f' % (i+1,loss_temp/len(train_loader)))
    test(model)
    torch.save(model.state_dict(), 'model_alexnet_params.pth')

def test(model):
    # 批量数目
    batch_size = 10
    # 预测正确个数
    correct = 0
    # 加载数据
    test_path = r"D:\python\data\flowers\net_test_images"
    test_set = My_Dataset_test(test_path, transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size, shuffle=False)
    # 开始
    for batch_data,batch_label in test_loader:
        # 放入GPU中
        batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
        # 预测
        prediction = model(batch_data)
        # 将预测值中最大的索引取出，其对应了不同类别值
        predicted = torch.max(prediction.data, 1)[1]
        # 获取准确个数
        correct += (predicted == batch_label).sum()
    print('准确率: %.2f %%' % (100 * correct / 500)) # 因为总共500个测试数据


if __name__ == '__main__':
    train()



