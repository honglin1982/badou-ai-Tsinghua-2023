import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
import torch.utils.data as Data


class MnistNet(torch.nn.Module):
    """ torch 网络层定义 """

    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)
        pass

    def forward(self, x: Tensor):
        """ 正向传播 """
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


class PyTorchBaseModel:
    """ 训练模型 """

    def __init__(self, net: MnistNet, cost_str=None, optimist_str=None):
        self.net = net
        self.cost = self.create_cost(cost_str)
        self.optimizer = self.create_optimizer(optimist_str)

    def create_cost(self, cost_str):
        # 损失函数
        support_cost = {
            "CROSS_ENTROPY": nn.CrossEntropyLoss(),
            "MSE": nn.MSELoss(),
        }
        return support_cost[cost_str]

    def create_optimizer(self, optimist_str, **rests):
        # 优化器包
        support_optimist = {
            "SGD": optim.SGD(self.net.parameters(), lr=0.1, **rests),
            "ADAM": optim.Adam(self.net.parameters(), lr=0.1, **rests),
            "RMSP": optim.RMSprop(self.net.parameters(), lr=0.1, **rests),
        }
        return support_optimist[optimist_str]

    def train(self, train_loader: torch.utils.data.DataLoader, epoch_nums=3):
        """
        1.先将梯度归零：
            一次正向传播得到预测值
            计算损失值
        2.反向传播得到每个参数的梯度值
        3.根据梯度进行参数更新
        """
        for epoch in range(epoch_nums):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # --input_data: torch.Size([32, 1, 28, 28]), labels: tensor([8, 8, 4, 4, 7, 1, 3, 5, 2, 7, 5, 3, 4, 3, 2, 4, 0, 9, 5, 9, 2, 8, 3, 8,
                #         6, 8, 6, 4, 8, 3, 4, 3])
                input_data, labels = data
                # 1.先将梯度归零: 目前主流的深度学习模型的优化器都是随机批次梯度下降法，即对一个batchsize数据去求平均梯度，根据得到的这个平均梯度去更新所有参数。因此，每个batchsize数据的梯度是唯一的，每次重新开始一个批次的计算必须先将参数之前的对应梯度清零。
                self.optimizer.zero_grad()
                # 一次正向传播得到预测值, 输出的是 32行数据，每行10列的概率值
                outputs = self.net(input_data)
                # 计算损失值
                loss = self.cost(outputs, labels)
                # 2.反向传播得到每个参数的梯度值
                loss.backward()
                # 3.根据梯度进行参数更新
                self.optimizer.step()
                running_loss += loss.item()  # .item() 在这里是提取数据
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' % (
                    epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print('Finished Training ...')

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict : todo 不用计算梯度
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)  # torch.Size([32, 10])
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def mnist_load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转化为tensor结构
        transforms.Normalize(mean=[0, ], std=[1, ])  # 归一化
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # 下载训练数据
    train_loader = Data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)  # 加载训练数据
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)  # 下载测试数据
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True, num_workers=2)  # 加载测试数据
    return train_loader, test_loader


if __name__ == '__main__':
    # 先创建神经网络模型
    m_net = MnistNet()
    model = PyTorchBaseModel(net=m_net, cost_str='CROSS_ENTROPY', optimist_str='RMSP')
    train_loader, test_loader = mnist_load_data()  # 加载数据
    model.train(train_loader)
    model.evaluate(test_loader)
    pass
