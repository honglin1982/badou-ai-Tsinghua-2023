import torch
import torch.nn as nn                           # 神经网络模块
import torch.optim as optim                     # 优化器有很多种，这里使用SGD
import torch.nn.functional as F                 # 激励函数都在这
import torchvision                              # 数据库模块
import torchvision.transforms as transforms     # 数据预处理模块

class Model:
    def __init__(self, net, cost, optimist):                # 构造函数
        self.net = net                                      # 网络
        self.cost = self.create_cost(cost)                  # 损失函数
        self.optimizer = self.create_optimizer(optimist)    # 优化器
        pass

    def create_cost(self, cost):                                            # 创建损失函数
        support_cost = {                                                    # 损失函数字典
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),                         # 交叉熵损失函数
            'MSE': nn.MSELoss()                                             # 均方误差损失函数
        }

        return support_cost[cost]                                           # 返回损失函数

    def create_optimizer(self, optimist, **rests):                          # 创建优化器
        support_optim = {                                                   # 优化器字典
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),       # SGD的参数：params, lr, momentum, dampening, weight_decay, nesterov，**rests表示可变参数, 此处表示momentum
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),    # ADAM的参数：params, lr, betas, eps, weight_decay, amsgrad,**rests表示可变参数, 此处表示betas,betas是用于计算梯度和梯度平方的运行平均值的系数
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)  # RMSP的参数：params, lr, alpha, eps, weight_decay, momentum, centered,**rests表示可变参数, 此处表示alpha, alpha是用于计算梯度平方的运行平均值的系数
        }

        return support_optim[optimist]                                      # 返回优化器, 优化器的参数是网络的参数

    def train(self, train_loader, epoches=3):                               # 训练
        for epoch in range(epoches):                                        # loop over the dataset multiple times
            running_loss = 0.0                                              # 每个epoch都要清零loss
            for i, data in enumerate(train_loader, 0):                      # enumerate(sequence, [start=0])，i序号，data是数据
                inputs, labels = data                                       # get the inputs; data is a list of [inputs, labels]

                self.optimizer.zero_grad()                                  # 将梯度参数归零

                # forward + backward + optimize                             # 前向传播+反向传播+优化
                outputs = self.net(inputs)                                  # 前向传播
                loss = self.cost(outputs, labels)                           # 计算损失
                loss.backward()                                             # 反向传播
                self.optimizer.step()                                       # 优化

                running_loss += loss.item()                                 # 损失累加
                if i % 100 == 0:                                            # 每100个batch打印一次， 一个batch是32张图片, 100个batch就是3200张图片
                    print('[epoch %d, %.2f%%] loss: %.3f' %                 # 打印每个epoch的loss
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100)) # i是第几个batch，len(train_loader)是总共有多少个batch
                    running_loss = 0.0                                      # 清零loss，为下一个epoch做准备

        print('Finished Training')                                          # 训练完成

    def evaluate(self, test_loader):                    # 测试
        print('Evaluating ...')                         # 评估
        correct = 0                                     # 统计正确的
        total = 0                                       # 总数
        with torch.no_grad():  # no grad when test and predict # 测试时不需要反向传播，所以不需要梯度
            for data in test_loader:                    # 遍历测试集
                images, labels = data                   # get the inputs; data is a list of [inputs, labels]

                outputs = self.net(images)              # forward
                predicted = torch.argmax(outputs, 1)    # get the index of the max log-probability
                total += labels.size(0)                 # labels.size(0)是labels这个tensor的行数，因为一行代表一个数据
                correct += (predicted == labels).sum().item() # 预测正确的总数

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total)) # 打印识别准确率

def mnist_load_data():                                  # 加载数据
    transform = transforms.Compose(                     # transforms.Compose就是将多个变换组合在一起
        [transforms.ToTensor(),                         # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
         transforms.Normalize([0,], [1,])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,                                # torchvision.datasets.MNIST是一个数据集，root是数据集的目录，train=True表示训练集，train=False表示测试集
                                            download=True, transform=transform)                     # transform是对数据进行变换，这里先转换为tensor，再归一化至[0-1]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,                              # util.data.DataLoader是一个比较通用的数据加载器。 批大小设置为32,即每次训练模型传入32张图片
                                              shuffle=True, num_workers=2)                          # 将shuffle置为True，打乱数据

    testset = torchvision.datasets.MNIST(root='./data', train=False,                                # 测试集
                                           download=True, transform=transform)                      # transform是对数据进行变换，这里先转换为tensor，再归一化至[0-1]
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)    # 将shuffle置为True，打乱数据
    return trainloader, testloader


class MnistNet(torch.nn.Module):                # 继承 torch 的 Module，pytorch 的神经网络模块化程度很高，只需要定义好 forward 函数，backward函数会在使用autograd时自动定义
    def __init__(self):
        super(MnistNet, self).__init__()        # 调用父类的构造函数，下面继承了父类的属性，torch.nn.Module具有持久化特性，即在类中定义的成员在反复调用中始终存在
        self.fc1 = torch.nn.Linear(28*28, 512)  # 定义全连接层，输入特征数为28*28，输出为512
        self.fc2 = torch.nn.Linear(512, 512)    # 定义全连接层，输入特征数为512，输出为512
        self.fc3 = torch.nn.Linear(512, 10)     # 定义全连接层，输入特征数为512，输出为10

    def forward(self, x):                       # 定义前向传播过程，输入为x
        x = x.view(-1, 28*28)                   # 将输入x展平为28*28
        x = F.relu(self.fc1(x))                 # 输入x经过全连接层fc1，再经过ReLU激活函数，然后赋值给x
        x = F.relu(self.fc2(x))                 # 输入x经过全连接层fc2，再经过ReLU激活函数，然后赋值给x
        x = F.softmax(self.fc3(x), dim=1)       # 输入x经过全连接层fc3，再经过Softmax激活函数，然后赋值给x
        return x

if __name__ == '__main__':
    # train for mnist
    net = MnistNet()                                # 实例化网络对象
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')     # 交叉熵损失函数，随机梯度下降优化器
    train_loader, test_loader = mnist_load_data()   # 加载数据，训练集和测试集
    model.train(train_loader)                       # 训练
    model.evaluate(test_loader)                     # 测试
