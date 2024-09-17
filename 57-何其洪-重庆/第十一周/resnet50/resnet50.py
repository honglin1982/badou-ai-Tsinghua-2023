# -*- coding: utf-8 -*-
import pathlib
import time
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset


class MyDataLoader(Dataset):
    def __init__(self, path, transform):
        self.transform = transform
        data_root = pathlib.Path(path)
        paths = list(data_root.glob('*.*'))
        self.all_image_paths = []
        self.all_image_labels = []
        for image_path in paths:
            self.all_image_paths.append(str(image_path))
            self.all_image_labels.append(0 if image_path.name.split('.')[0] == 'cat' else 1)

    def __getitem__(self, index):
        img = Image.open(self.all_image_paths[index]).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(self.all_image_labels[index])
        return img, label

    def __len__(self):
        return len(self.all_image_paths)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, places,
                 stride=1, is_down_sample=False, expansion=4):
        super().__init__()
        self.is_down_sample = is_down_sample
        self.expansion = expansion

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, places, 1, 1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(places, places, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(places, places * self.expansion, 1, 1, bias=False),
            nn.BatchNorm2d(places * self.expansion)
        )

        if is_down_sample:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, places * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        if self.is_down_sample:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.expansion = 4

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1) # 输出 64, 56, 56
        )
        self.layer1 = self.make_layer(64, 64, 3, 1)
        self.layer2 = self.make_layer(256, 128, 4, 2)
        self.layer3 = self.make_layer(512, 256, 6, 2)
        self.layer4 = self.make_layer(1024, 512, 3, 2)

        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_channels, places, block_num, stride=1):
        layers = [Bottleneck(in_channels, places, stride, True, self.expansion)]
        block_num -= 1
        for i in range(block_num):
            layers.append(Bottleneck(places * self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def train():
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.RandomResizedCrop(size=224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
    path = r'D:\BaiduNetdiskDownload\【11】CNN&图像识别\代码\Alexnet\train'
    train_loader = DataLoader(MyDataLoader(path, train_transform), 32, True, num_workers=1)
    net = ResNet50()
    print(net)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.8, 3, verbose=True, min_lr=0.0001)

    net.cuda()
    loss_func.cuda()
    total_train_step = 0
    running_loss = 0.0
    for epoch in range(50):
        for i, data in enumerate(train_loader):
            start_time = time.time()
            train_data, train_labels = data
            train_data = train_data.cuda()
            train_labels = train_labels.cuda()
            result = net(train_data)
            loss = loss_func(result, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_train_step += 1
            if total_train_step % 100 == 0:
                now_loss = running_loss / 100
                scheduler.step(now_loss)
                running_loss = 0.0
                print('epoch: {}, 耗时：{}'.format(epoch, time.time() - start_time))
                print("训练次数: {}, Loss: {}, 平均损失: {}, 当前lr: {}".format(total_train_step, loss, now_loss,
                                                                                optimizer))

    torch.save(net, 'cat-dot.pt')
    print('模型已保存')


def evaluate(net, loader):
    right_num = 0
    total = 0
    # 不保存梯度
    with torch.no_grad():
        for data in loader:
            test_data, test_labels = data
            test_data = test_data.cuda()
            test_labels = test_labels.cuda()
            # 计算
            start_time = time.time()
            outputs = net(test_data)
            print("推理耗时：", (time.time() - start_time) / len(test_data))
            # 获取每个最大值的索引
            predicted = torch.argmax(outputs, 1)
            total += len(test_labels)
            # 取出正确数量
            right_num += (predicted == test_labels).sum().item()
    print('正确率：', right_num/total)


def test():
    net = torch.load('cat-dot.pt')
    net.cuda()
    net.eval()
    img = Image.open(r"C:\Users\MECHREV\Desktop\th.jpg").convert('RGB').resize(
        (224, 224))
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
    img_ = test_transform(img).unsqueeze(0)
    img_ = img_.cuda()
    class_names = ['cat', 'dog']
    out = net(img_)
    print(out)
    _, indices = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    print(percentage)
    perc = percentage[int(indices)].item()
    result = class_names[indices]
    print('predicted:', result)


if __name__ == '__main__':
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.RandomResizedCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
    path = r'D:\BaiduNetdiskDownload\【11】CNN&图像识别\代码\Alexnet\train'
    train_loader = DataLoader(MyDataLoader(path, test_transform), 32, True, num_workers=1)

    net = torch.load('cat-dot.pt')
    net.cuda()
    net.eval()
    evaluate(net, train_loader)
