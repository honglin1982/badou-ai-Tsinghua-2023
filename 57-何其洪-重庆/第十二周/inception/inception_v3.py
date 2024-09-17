# -*- coding: utf-8 -*-
import pathlib
import time

import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader


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


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, 1, 1, 0)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, 1, 1, 0)
        self.branch5x5_2 = BasicConv2d(48, 64, 5, 1, 2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, 1, 1, 0)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, 3, 1, 1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, 3, 1, 1)

        self.branch_avg_pool = nn.AvgPool2d(3, 1, 1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, 1, 1, 0)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = self.branch_avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        # 某个维度进行拼接，pytorch的tensor是(n, c, h, w)的形状，dim=1 就是延c(通道)方向拼接
        return torch.cat([branch1x1, branch5x5, branch3x3dbl, branch_pool], dim=1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, 3, 2, 0)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, 1, 1, 0)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, 3, 1, 1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, 3, 2, 0)

        self.branch_pool = nn.MaxPool2d(3, 2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = self.branch_pool(x)
        return torch.cat([branch3x3, branch3x3dbl, branch_pool], 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channel7x7):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, 1, 1, 0)

        self.branch7x7_1 = BasicConv2d(in_channels, channel7x7, 1, 1, 0)
        self.branch7x7_2 = BasicConv2d(channel7x7, channel7x7, (1, 7), 1, (0, 3))
        self.branch7x7_3 = BasicConv2d(channel7x7, 192, (7, 1), 1, (3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, channel7x7, 1, 1, 0)
        self.branch7x7dbl_2 = BasicConv2d(channel7x7, channel7x7, (7, 1), 1, (3, 0))
        self.branch7x7dbl_3 = BasicConv2d(channel7x7, channel7x7, (1, 7), 1, (0, 3))
        self.branch7x7dbl_4 = BasicConv2d(channel7x7, channel7x7, (7, 1), 1, (3, 0))
        self.branch7x7dbl_5 = BasicConv2d(channel7x7, 192, (1, 7), 1, (0, 3))

        self.branch_avg_pool = nn.AvgPool2d(3, 1, 1)
        self.branch_pool = BasicConv2d(in_channels, 192, 1, 1, 0)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = self.branch_avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)
        return torch.cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], dim=1)


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, 1, 1, 0)
        self.branch3x3_2 = BasicConv2d(192, 320, 3, 2, 0)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, 1, 1, 0)
        self.branch7x7x3_2 = BasicConv2d(192, 192, (1, 7), 1, (0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, (7, 1), 1, (3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, 3, 2, 0)

        self.branch_pool = nn.MaxPool2d(3, 2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = self.branch_pool(x)
        return torch.cat([branch3x3, branch7x7x3, branch_pool], dim=1)


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, 1, 1, 0)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, 1, 1, 0)
        self.branch3x3_2a = BasicConv2d(384, 384, (1, 3), 1, (0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, (3, 1), 1, (1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, 1, 1, 0)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, 3, 1, 1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, (1, 3), 1, (0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, (3, 1), 1, (1, 0))

        self.branch_avg_pool = nn.AvgPool2d(3, 1, 1)
        self.branch_pool = BasicConv2d(in_channels, 192, 1, 1, 0)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3_a = self.branch3x3_2a(branch3x3)
        branch3x3_b = self.branch3x3_2b(branch3x3)
        branch3x3 = torch.cat([branch3x3_a, branch3x3_b], dim=1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl_a = self.branch3x3dbl_3a(branch3x3dbl)
        branch3x3dbl_b = self.branch3x3dbl_3b(branch3x3dbl)
        branch3x3dbl = torch.cat([branch3x3dbl_a, branch3x3dbl_b], dim=1)
        branch_pool = self.branch_avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)
        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], dim=1)


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            BasicConv2d(3, 32, 3, 2, 0),
            BasicConv2d(32, 32, 3, 1, 0),
            BasicConv2d(32, 64, 3, 1, 1),

            nn.MaxPool2d(3, 2),

            BasicConv2d(64, 80, 1, 1, 0),
            BasicConv2d(80, 192, 3, 1, 0),

            nn.MaxPool2d(3, 2)
        )

        self.block1_1 = InceptionA(192, 32)
        self.block1_2 = InceptionA(256, 64)
        self.block1_3 = InceptionA(288, 64)

        self.block2_1 = InceptionB(288)
        self.block2_2 = InceptionC(768, 128)
        self.block2_3 = InceptionC(768, 160)
        self.block2_4 = InceptionC(768, 160)
        self.block2_5 = InceptionC(768, 192)

        self.block3_1 = InceptionD(768)
        self.block3_2 = InceptionE(1280)
        self.block3_3 = InceptionE(2048)

        self.avg_pool = nn.AvgPool2d(8)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        # h  w  c
        # 299 299 3 -> 35 35 192
        x = self.conv1(x)
        # 35 35 192 -> 35 35 256
        x = self.block1_1(x)
        # 35 35 256 -> 35 35 288
        x = self.block1_2(x)
        # 35 35 288 -> 35 35 288
        x = self.block1_3(x)
        # 35 35 288 -> 17 17 768
        x = self.block2_1(x)
        # 17 17 768 -> 17 17 768
        x = self.block2_2(x)
        # 17 17 768 -> 17 17 768
        x = self.block2_3(x)
        # 17 17 768 -> 17 17 768
        x = self.block2_4(x)
        # 17 17 768 -> 17 17 768
        x = self.block2_5(x)
        # 17 17 768 -> 8 8 1280
        x = self.block3_1(x)
        # 8 8 1280 -> 8 8 2048
        x = self.block3_2(x)
        # 8 8 2048 -> 8 8 2048
        x = self.block3_3(x)
        # 8 8 2048 -> 1 1 2048
        x = self.avg_pool(x)
        # 1 1 2048 -> 1 1 2048
        x = self.dropout(x)
        # 1 1 2048 -> 2048
        x = x.view(x.size(0), -1)
        # 2048 -> 2
        return self.fc(x)


if __name__ == '__main__':
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(299),
        torchvision.transforms.RandomResizedCrop(size=299),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
    path = r'D:\BaiduNetdiskDownload\【11】CNN&图像识别\代码\Alexnet\train'
    train_loader = DataLoader(MyDataLoader(path, train_transform), 32, True, num_workers=1)
    net = InceptionV3()
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
