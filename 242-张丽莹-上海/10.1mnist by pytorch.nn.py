# 在pytorch框架内，搭建神经网络实现手写数字识别

import torch
import torch.nn as nn
import torch.optim as optim      # 优化器模块
import torch.nn.functional as F  # functional模块，适用于加载各种现成函数，且不需要实例化
import torchvision               # torchvision模块，用于数据处理、模型使用和结果可视化多种功能
import torchvision.transforms as transforms    # 图像和视频的预处理操作，如缩放、裁剪、翻转、标准化等


# 数据加载
def mnist_load_data():   # 此处load data不需要传入参数
    # transforms.Compose函数: 将多个数据转换操作组合起来
    # transforms.ToTensor: 将图像数据从PIL图像或NumPy数组转换为PyTorch张量
    # transforms.Normalize(均值序列, 标准差序列)—: 归一化
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])

    # 创建一个训练数据集的实例:torchvision.datasets.库内置数据集
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    # 创建一个用于训练的数据加载器（DataLoader）对象
    # torch.utils.data.Dataloader函数
    # numworkers是进程数量，2是两个，一个处理当下一个处理下一批，默认0是只主进程，如果系统资源允许，可以增加。
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    # 对测试集再来一遍
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                              shuffle=True, num_workers=2)

    return trainloader, testloader

# 类1：模型的参数、方法
class Model:
    # 模型的构造函数（网络、损失函数、优化器）
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    # 损失函数选择：以字典形式呈现以做选择
    # 交叉熵（适用于分类问题）、MSE均方误差。
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    # 优化器选择：以字典形式呈现以做选择
    # SGD：梯度下降优化法，学习率x梯度来更新参数，学习率固定要手动设置，较为稳定
    # Adam：自动调整学习率，具有较快的收敛速度。RMSP和Adam都是自适应学习率优化器
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    # 训练函数：训练数据加载器，迭代次数epoches
    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0

            # 遍历训练数据加载器，获取每个批次的数据和标签。
            for i, data in enumerate(train_loader, 0):
                # 每个batch都要来一套如下流程：
                # 获取数据、标签——梯度清零——算输出——计算损失——反向传播——更新参数

                inputs, labels = data  # 张量数据
                self.optimizer.zero_grad()
                # pytorch特性是不清零计算，所以得将优化器中的梯度缓存清零，以便进行反向传播计算新的梯度。
                outputs = self.net(inputs)           # net函数在哪呢？
                loss = self.cost(outputs, labels)    # 因其内置损失函数的参数是output和label
                loss.backward()                      # backward是内置反向传播函数，计算梯度
                self.optimizer.step()                # 根据梯度更新模型的参数。

                running_loss += loss.item()          # loss.item只能获取包含单个值的张量的数值

                if i % 100 == 0: # 求余数为0，就是每满100次算一次epoch
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss/100))
                          # [epoch 1, 0.00%] loss: 0.023
                          # 整数%d:epoch + 1 指当前是第几轮
                          # 浮点数百分比%.2f%%:(i + 1)*1./len(train_loader) 当前batch数在整个训练集中所占的百分比，即训练进度。
                          # 浮点数%.3f：running_loss/100  当前训练的平均损失值
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # 训练集不需要计算梯度和更新参数
            for data in test_loader:
                images, labels = data   # 每个data都有数据和标签

                outputs = self.net(images)                      # 这个net？
                predicted = torch.argmax(outputs, 1)            # 1表示在第一个维度上寻找最大值，即每个样本的预测得分
                total += labels.size(0)                         # 计算测试集的总样本数
                correct += (predicted == labels).sum().item()   # item函数同上，提取张量里的单一数值

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


# 类2：模型的网络结构
class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()  #?
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

# 主函数运行
if __name__ == '__main__':
    net = MnistNet()  #
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')  # 在类1构造函数里就定了这仨
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
