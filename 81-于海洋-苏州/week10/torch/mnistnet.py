import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    @staticmethod
    def create_cost(cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        #
        """
        self.net.parameters()：这是一个传递给优化器的参数列表，parameters() 方法用于获取模型中所有可学习的参数。
        lr=0.001：学习率（learning rate）是控制参数更新步幅的超参数，表示每次参数更新的幅度大小。这里设置学习率为 0.001。
        **rests：这是额外的参数，用于传递其他可选的优化器参数，例如权重衰减（weight decay）、动量（momentum）等。**rests 表示将额外的参数以关键字参数的形式传递给 RMSprop 优化器。
        """
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' % (epoch + 1, (i + 1) * 1. / len(loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict
            for data in loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def gen_data():
    # 把图片转换成张量，并把像素标准化到[0-1]范围内
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0, ], [1, ])])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_l = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_l = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True, num_workers=2)
    return train_l, test_l


if __name__ == '__main__':
    # train for mnist
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = gen_data()
    model.train(train_loader)
    model.evaluate(test_loader)
