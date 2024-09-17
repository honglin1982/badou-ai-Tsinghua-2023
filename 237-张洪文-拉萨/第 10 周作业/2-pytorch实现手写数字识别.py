import torch  # PyTorch的核心库，用于构建和训练深度神经网络
import torch.nn as nn  # 构建神经网络所需的各种模块和函数。
import torch.optim as optim  # 提供了各种优化算法的实现，用于在训练神经网络时更新模型的参数
import torch.nn.functional as F  # 提供了一些非常常用的函数，用于构建神经网络的各种层和操作
import torchvision  # 用于处理计算机视觉任务的数据集和图像处理
import torchvision.transforms as transforms  # 提供了一系列图像预处理的工具

# 构建神经网络结构：用于在MNIST手写数字数据集上进行图像分类
class MnistNetwork(nn.Module):   # 继承自 torch.nn.Module 类
    def __init__(self):  # 初始化网络结构
        super(MnistNetwork, self).__init__()  # 调用父类的构造方法
        self.fc1 = torch.nn.Linear(28*28, 512)  # 第1个全连接层（线性层）接收28*28维的输入，输出512维的特征
        self.fc2 = torch.nn.Linear(512, 512)  # 第2个全连接层，输入512 输出512
        self.fc3 = torch.nn.Linear(512, 10)  # 第3个全连接层，接收512 输出10.z这个层的输出将用于分类任务，因为MNIST数据机中有10个数字类别（0-9）

    def forward(self, x):  # 定义前向传播的方法，在这里指定数据在神经网络中如何进行前向传递
        x = x.view(-1, 28*28)  # 将输入张量 x 的形状从（batch_size,1,28,28）转换为（batch_size,28*28）.-1表自动计算batch_size的维度
        x = F.relu(self.fc1(x))  # 第1层的激活函数，使用relu函数对其输出进行非线性变换
        x = F.relu(self.fc2(x))  # 第2层的激活函数，使用relu函数对其输出进行非线性变换
        x = F.softmax(self.fc3(x), dim=1)  # 第3层的激活函数，使用softmax函数将输出转换到0~1之间的概率分布，dim=1表示沿着第一个维度计算softmax

        return x  # 返回经过神经网络处理后的输出张量，用于训练和测试分类任务

# 加载 MNIST 数据集：返回用于训练和测试的数据加载器（DataLoader）
def mnist_data_load():
    # 定义数据预处理管道：用于对图像进行转换和归一化
    transform = transforms.Compose(  # 定义对象，将一系列图像操作组合在一起
        [transforms.ToTensor(),  # 将图像转换为PyTorch张量
         transforms.Normalize([0, ], [1, ])])  # 进行图像归一化
    # 加载训练集
    train_set = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(  # 训练用的数据加载器
        train_set, batch_size=32, shuffle=True, num_workers=2)
    # 加载测试集
    test_set = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(  # 测试的数据加载器
        test_set, batch_size=32, shuffle=True, num_workers=2)  # 数据 批次 打乱 进程

    return trainloader, testloader  # 返回加载器

# 定义模型
class MnistModel:
    # 初始化：接收 神经网络模型  损失函数名称  优化器名称
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)

    # 传入损失函数名称 cost 创建对应的损失函数： 损失函数用于衡量模型的输出与真实标签之间的差异
    def create_cost(self, cost):
        support_cost = {
            "CROSS_ENTROPY": nn.CrossEntropyLoss(),  # 交叉熵损失函数
            "MSE": nn.MSELoss()  # 均方误差损失函数
        }
        return support_cost[cost]

    # 传入优化器名称 optimist 创建对应的优化器：优化器用于更新神经网络模型的参数，以便在训练过程中优化损失函数
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            "SGD": optim.SGD(self.net.parameters(), lr=0.1, **rests),  # 随机梯度下降优化器
            "ADAM": optim.Adam(self.net.parameters(), lr=0.01, **rests),  # Adam优化器
            "RMSP": optim.RMSprop(self.net.parameters(), lr=0.001, **rests)  # RMSprop优化器
        }
        return support_optim[optimist]

    # 用于训练神经网络模型
    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print("[epoch %d, %.2f%%] loss: %.3f" % (epoch+1, ((i+1)*1./len(train_loader))*100, running_loss/100))
        print("Finished Training !!!")

    # 用于评估神经网络模型在测试集上的准确率
    def evaluate(self, test_loader):
        print("Evaluating ...")
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"网络对测试图像的的准确性: {100 * correct/total}%")


if __name__ == '__main__':
    # 加载数据集：获取数据加载器
    train_loader, test_loader = mnist_data_load()
    network = MnistNetwork()  # 网络
    module = MnistModel(network, "CROSS_ENTROPY", "RMSP")  # 模型
    module.train(train_loader)  # 训练
    module.evaluate(test_loader)  # 推理

