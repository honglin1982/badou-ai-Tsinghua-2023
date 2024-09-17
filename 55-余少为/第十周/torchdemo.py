import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

batch_size = 32
learning_rate = 0.001
epoches = 3

# 下载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
# 数据装载到DateLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def train(model):
    model.train()
    # 设置误差函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    for epoch in range(epoches):
        running_acc = 0.0
        running_loss = 0.0

        for i, data in enumerate(train_loader, 1):
            img, label = data
            if use_gpu:
                img = img.cuda()
                label = label.cuda()

            # 正向传播
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item()
            _, pred = torch.max(out, 1)
            running_acc += (pred == label).sum()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:    # 每100个batch，观测一下训练情况
                print(f'Epoch [{epoch+1}/{epoches}], Step [{i}/{len(train_loader)}], '
                      f'Loss: {running_loss/(batch_size*100):.4f}, Acc: {running_acc/batch_size:.4f} %')
                running_acc = 0.0
                running_loss = 0.0


def test(model):
    n_correct = 0
    n_samples = 0
    model.eval()

    with torch.no_grad():   # 测试和预测时不需要计算梯度
        for data in test_loader:
            img, label = data
            if use_gpu:
                img = img.cuda()
                label = label.cuda()

            out = model(img)
            _, pred = torch.max(out, 1)
            n_samples += label.size(0)
            n_correct += (pred == label).sum()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the {n_samples} test images: {acc:.4f} %')


if __name__ == '__main__':
    model = MnistNet()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    train(model)
    test(model)
