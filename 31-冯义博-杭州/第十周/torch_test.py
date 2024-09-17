import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class Model:
    def __init__(self, net):
        self.net = net
        self.cost = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=0.001)

    def train(self, train_loader, epochs=3):
        for epoch in range(epochs):
            running_loss = 0.0
            # 遍历并设置下标
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                # 每一个batch梯度归零 损失函数不替换 累加
                self.optimizer.zero_grad()
                # forward
                outputs = self.net(inputs)
                # 使用MSE损失函数 需要对labels进行one hot编码 保证tensor维度相同
                loss = self.cost(outputs.float(), F.one_hot(labels, num_classes=10).float())
                loss.requires_grad_(True)
                loss.backward()
                # 更新所有参数
                self.optimizer.step()
                # 累加损失函数（无用 纯统计）
                # running_loss += loss.item()
                if i % 100 == 0:
                    p = (i + 1) * 100. / len(train_loader)
                    # 输出处理进度
                    print(f"epoch = {epoch} , handle {p}%")
        print('Finished Training')

    def evaluate(self, test_loader):
        print("predict starting...")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                inputs, labels = data
                outputs = self.net(inputs)
                result = torch.argmax(outputs, 1)
                total += len(labels)
                correct += (result == labels).sum().item()

        accuracy = correct * 100. / total
        print(f"accuracy = {accuracy}%")


class MnistNet(nn.Module):

    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0, ], [1, ])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader


if __name__ == "__main__":
    net = MnistNet()
    model = Model(net)
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
