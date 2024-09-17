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


class Moblienet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(True)
        )
        self.conv_dw1 = self.depthwise_conv_block(32, 64, 2)
        self.conv_dw2 = self.depthwise_conv_block(64, 128, 2)
        self.conv_dw3 = self.depthwise_conv_block(128, 128, 1)
        self.conv_dw4 = self.depthwise_conv_block(128, 256, 2)
        self.conv_dw5 = self.depthwise_conv_block(256, 256, 1)
        self.conv_dw6 = self.depthwise_conv_block(256, 512, 2)

        self.conv_dw7 = self.depthwise_conv_block(512, 512, 1)
        self.conv_dw8 = self.depthwise_conv_block(512, 512, 1)
        self.conv_dw9 = self.depthwise_conv_block(512, 512, 1)
        self.conv_dw10 = self.depthwise_conv_block(512, 512, 1)
        self.conv_dw11 = self.depthwise_conv_block(512, 512, 1)

        self.conv_dw12 = self.depthwise_conv_block(512, 1024, 2)
        self.conv_dw13 = self.depthwise_conv_block(1024, 1024, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_dw1(x)
        x = self.conv_dw2(x)
        x = self.conv_dw3(x)
        x = self.conv_dw4(x)
        x = self.conv_dw5(x)
        x = self.conv_dw6(x)
        x = self.conv_dw7(x)
        x = self.conv_dw8(x)
        x = self.conv_dw9(x)
        x = self.conv_dw10(x)
        x = self.conv_dw11(x)
        x = self.conv_dw12(x)
        x = self.conv_dw13(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def depthwise_conv_block(self, in_channels, filters, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, bias=False, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, filters, 1, 1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU6(filters)
        )


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
    net = Moblienet()
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
        torch.save(net, 'cat-dto-{}.pt'.format(epoch))

    torch.save(net, 'cat-dot.pt')
    print('模型已保存')
