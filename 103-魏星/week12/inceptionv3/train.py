# _*_ coding : utf-8 _*_
# @Time : 2023/7/24 13:37
# @Author : weixing
# @FileName : alexnet
# @Project : cv

import os
import json
import sys

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from InceptionV3 import InceptionV3


BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断是部署在GPU上还是在CPU上
num_workers = 0  # 给Dataloader设置worker数量
EPOCH = 3  # 训练数据集的轮次
best_acc = 0.87 #最低准确率，超过这个值则保存模型
Model_File_Url = "./torch-inceptionV3.h5"  #模型保存路径

def saveTrainDataTags(train_dataset):
    labels_list = train_dataset.class_to_idx  # class_to_idx就是获取train_dataset下每个文件夹的名称，并按字典返回

    cla_dict = dict((val, key) for key, val in labels_list.items())  # 将label的键值对反过来

    # write dict into json file，方便预测的时候调用
    json_str = json.dumps(cla_dict, indent=4)
    with open('cifar10_class.json', 'w') as json_file:
        json_file.write(json_str)

def load_data():
    # 数据预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(299),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "test": transforms.Compose([transforms.Resize((299, 299)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    # 先执行down&saveCifar10.py 将图片分类为 train-标签-图片，test-标签-图片
    image_path="D:\\ProgramData\\project-data\cv\\11-lesson11\\cifar_10_data"

    # 创建数据集（打包数据集）
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # 创建验证集（打包验证集）
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["test"])
    test_num = len(test_dataset)

    saveTrainDataTags(train_dataset)

    # 加载训练集
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers)

    # 加载测试集
    test_loader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=num_workers)

    print("using {} images for training, {} images for test.".format(train_num, test_num))
    return train_loader, test_loader

class Model():
    def __init__(self, network):
        self.network = network
        # 定义损失函数、优化器
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.045)

    def do_train(self, epochs, train_loader):
        print("start training")
        # 训练模型
        for epoch in range(epochs):
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)  # 进度条,（iterable=可迭代对象，file: 输出指向位置, 默认是终端, 一般不需要设置）
            for i, (inputs, labels) in enumerate(train_bar):
                self.optimizer.zero_grad()
                # print(inputs.shape, labels.shape)
                outputs = self.network(inputs.to(DEVICE))
                # print(outputs.shape)
                loss = self.loss_function(outputs, labels.to(DEVICE))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                train_bar.desc = "train epoch[{}/{}] loss:{:.4f}".format(epoch + 1, epochs, loss)
            # print('train;Epoch[{}/{}] Loss: %.4f'.format(epoch + 1, epochs, running_loss / len(train_loader)))
        print("train end")

    def do_eval(self, test_loader):
        print("start test")
        # 测试模型
        correct = 0
        total = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for inputs, labels in test_bar:
                outputs = self.network(inputs.to(DEVICE))
                _, predicted = torch.max(outputs.data, 1)
                correct += torch.eq(predicted, labels.to(DEVICE)).sum().item()
                total += labels.size(0)
            print("predict;测试集总个数 %d, 预测正确的个数 %d, 准确率 %.4f" % (total, correct, correct / total))

            # 保存参数
            if correct/total > best_acc:
                print("model to be saved")
                torch.save(self.network.state_dict(), Model_File_Url)

        print("test end")


def main():
    # 加载数据集
    train_loader, test_loader = load_data()
    # 网络
    network = InceptionV3(num_classes=10)
    network.to(DEVICE)  # 放到GPU上
    # 模型
    model = Model(network)
    # 训练
    epoch = 8
    model.do_train(epoch, train_loader)
    # 测试
    model.do_eval(test_loader)

if __name__=='__main__':
    main()

