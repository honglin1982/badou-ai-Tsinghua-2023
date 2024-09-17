# -*- coding: utf-8 -*-
import pathlib
import time

import torch
import torch.nn as nn
import torchvision
from PIL import Image
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


class VGG16Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)，再2X2最大池化，输出net为(112,112,64)
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)，再2X2最大池化，输出net为(56,56,128)
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)，再2X2最大池化，输出net为(28,28,256)。
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(28,28,512)，再2X2最大池化，输出net为(14,14,512)。
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(14,14,512)，再2X2最大池化，输出net为(7,7,512)
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 使用卷积方式模拟全连接层，经测试单次训练时长在28s - 30s，略高于使用全连接层的26s
            # nn.Conv2d(512, 4096, 7),
            # nn.Dropout(0.5),
            # nn.Conv2d(4096, 4096, 1),
            # nn.Dropout(0.5),
            # nn.Conv2d(4096, 2, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.RandomResizedCrop(size=224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
    path = r'D:\BaiduNetdiskDownload\【11】CNN&图像识别\代码\Alexnet\train'
    train_loader = DataLoader(MyDataLoader(path, train_transform), 64, True, num_workers=2)
    net = VGG16Net()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.8, 3, verbose=True, min_lr=0.0001)

    # net.cuda()
    # loss_func.cuda()
    total_train_step = 0
    running_loss = 0.0
    for epoch in range(50):
        for i, data in enumerate(train_loader):
            start_time = time.time()
            train_data, train_labels = data
            # train_data = train_data.cuda()
            # train_labels = train_labels.cuda()
            result = net(train_data)
            loss = loss_func(result, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_train_step += 1
            if total_train_step % 2 == 0:
                now_loss = running_loss / 2
                scheduler.step(now_loss)
                running_loss = 0.0
                print('epoch: {}, 耗时：{}'.format(epoch, time.time() - start_time))
                print("训练次数: {}, Loss: {}, 平均损失: {}, 当前lr: {}".format(total_train_step, loss, now_loss,
                                                                                optimizer))

    torch.save(net, 'cat-dot.pt')
    print('模型已保存')