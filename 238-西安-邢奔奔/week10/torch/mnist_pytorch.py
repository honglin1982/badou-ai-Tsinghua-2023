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

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()
                '''将之前计算的参数梯度归零，然后调用loos.backward()来进行反向传播
                然后计算梯度，在根据self.optimizer.step()根据优化算法来更新模型参数'''
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                        (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print('Traing Finish')


    def evaluate(self, test_loader):
        print('Evaluating...')
        correct = 0
        total = 0
        with torch.no_grad():#表示在这段代码中禁用梯度追踪和自动微分
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs,1)
                total += labels.size(0)#取数据的时候划分了batch_size
                correct += (predicted == labels).sum().item()
                ''''(predicted == labels).sum().item() 判断预测结果 predicted 是否与真实标签 labels 相匹配，并将匹配的数量进行累加。
                .sum().item() 的作用是将张量中匹配数量的值从张量中提取出来，以便进行后续的累加计算
                然后，.sum() 方法对布尔张量进行求和操作，将 True 视为 1，False 视为 0。这将得到一个表示正确预测数量的张量。
                最后，.item() 方法将这个张量中的值提取为一个 Python 标量（scalar），以便将其添加到变量 correct 中'''
        print('Accuracy of the network on the test images:%d %%' % (100 * correct / total))




def mnist_load_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0,], [1,])])
    '''transforms.ToTensor() 将 PIL 图像或 NumPy 数组转换为张量。它会按照规范化的方式将图像数据转换为浮点数张量，并将像素值从范围 [0, 255] 归一化到范围 [0, 1]。
        transforms.Normalize([0,], [1,]) 对张量进行规范化操作。它会将每个通道的张量数值减去均值并除以标准差，实现标准化（即使数据分布具有零均值和单位方差）。'''
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)

    return trainloader, testloader


class Mnistnet(torch.nn.Module):
    def __init__(self):
        super(Mnistnet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    '''构造好网络之后再将网络传到模型里，之后就可以进行推理和训练了'''
    net = Mnistnet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
