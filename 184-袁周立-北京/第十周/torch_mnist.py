import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torch.utils.data import DataLoader

'''
tf实现简单神经网络
pytorch实现手写数字识别
'''


class NetWork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NetWork, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, y=None):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x if y is None else F.cross_entropy(x, y)


class Model:
    def __init__(self, network, optim="SGD", lr=0.01):
        self.network = network
        self.optim = self.optim_chose(optim, lr)

    def optim_chose(self, optim, lr):
        if optim == "SGD":
            return torch.optim.SGD(self.network.parameters(), lr=lr)

    def train(self, train_data, epoch=5):
        print("开始训练...")
        for e in range(epoch):
            print("epoch %d begin：" % (e + 1))
            loss_sum = 0.0
            for i, data in enumerate(train_data):
                x, label = data
                x = x.view(x.shape[0], -1)

                self.optim.zero_grad()
                loss = self.network(x, label)
                loss.backward()
                self.optim.step()

                loss_sum += loss.item()
                if i % 100 == 0:
                    print("[epoch %d %.2f%%] loss: %.4f" % (e + 1, (i+1) * 100.0 / len(train_data), loss_sum / 100))
                    loss_sum = 0.0

    def evaluate(self, test_data):
        with torch.no_grad():
            correct = 0
            total = 0
            for i, data in enumerate(test_data):
                inputs, labels = data
                inputs = inputs.view(inputs.shape[0], -1)
                prediction = self.network(inputs)
                prediction = torch.argmax(prediction, dim=1)
                correct += torch.sum(prediction == labels).item()
                total += prediction.shape[0]
            acc = correct / total
            print("acc %.2f%%" % (acc * 100))


def loader_data(batch_size=32):
    transform = torchvision.transforms.ToTensor()

    # 直接读进来是个PIL.Image对象，需要转成tensor
    train_mnist = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_mnist = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    train_data = DataLoader(train_mnist, batch_size=batch_size, shuffle=False)
    test_data = DataLoader(test_mnist, batch_size=batch_size, shuffle=True)
    return train_data, test_data


def main():
    model = Model(NetWork(28 * 28, 500, 10), lr=0.1)
    train_data, test_data = loader_data()
    model.train(train_data)
    model.evaluate(test_data)


if __name__ == '__main__':
    main()
