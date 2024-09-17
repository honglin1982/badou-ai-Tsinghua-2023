import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

#构造网络
class MnistNet(nn.Module):

    def __init__(self):
        super(MnistNet, self).__init__() #调用父类初始化函数
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):

        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

#读入数据
def mnistnet_load_data():

    #装配数据类型转换和归一化函数
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0,],[1,])])
    #读入训练集 测试集 打乱 并 分批次
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=trans)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=trans)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4)
    return trainloader, testloader

#构建模型
class Model:

    def __init__(self, net, cost, opt):

        self.net = net;
        self.cost = self.chooseCost(cost)
        self.optimimer = self.chooseOptim(opt)

    def chooseCost(self, cost):

        supportLoss = {

            'CROSS_TROPY' : nn.CrossEntropyLoss(),
            'MSE' : nn.MSELoss()
        }
        return supportLoss[cost]

    def chooseOptim(self, opt):

        supportOptim = {

            'SGD' : optim.SGD(self.net.parameters(), lr=0.1),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001)
        }
        return supportOptim[opt]
    #训练
    def train(self, trainLoader, epoches=3):

        for epoch in range(epoches):

            runningLoss = 0.0
            for i, data in enumerate(trainLoader, 0):

                inputs, labels = data
                self.optimimer.zero_grad() #初始化优化函数梯度为0
                outputs = self.net(inputs) #forward
                loss = self.cost(outputs, labels)
                loss.backward()            #backward
                self.optimimer.step()      #单次优化

                runningLoss += loss.item()
                #实时显示进度
                if i % 100 == 0 :
                    print('[epoch %d, process %.2f%%] loss:%.3f'%(epoch + 1, 100.0 * (i + 1) / len(trainLoader), runningLoss / 100))
                    runningLoss = 0.0

    #评估
    def evaluate(self, testLoader):

        print('evaluating......')
        correct, total = 0, 0
        with torch.no_grad():
            for data in testLoader:

                inputs, labels = data
                outputs = self.net(inputs)
                predication = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predication == labels).sum()
            print('Accuracy of the net on the test images: %.2f%%'%(correct * 100.0 / total))

if __name__ == '__main__':

    mnistnet = MnistNet()
    trainLoader, testLoader = mnistnet_load_data()
    model = Model(mnistnet, 'CROSS_TROPY', 'RMSP')
    model.train(trainLoader)
    model.evaluate(testLoader)