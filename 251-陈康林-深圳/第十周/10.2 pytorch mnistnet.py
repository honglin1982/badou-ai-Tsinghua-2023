import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Model:
    def __init__(self,net,cost,optimst) :
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimst)
        pass

    def create_cost(self,cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    def create_optimizer(self,optimist,**rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.001, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(),lr= 0.001,**rests)
        }
        return support_optim[optimist]

    def train(self,tran_loader,epoches=3):
        running_loss = 0
        for epoch in range(epoches):
            for i,data in enumerate(tran_loader,0):
                inputs,labels = data

                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs,labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i%100 == 0:
                    print('[epoch%d,%.2f%%] loss:%.3f'%(epoch+1,(i+1)*1./len(tran_loader),running_loss/100))
                    running_loss = 0.0
        print('Finish tranning')
  
    def evaluate(self,test_loader):
        print('Evaluating...')
        correct = 0
        total = 0
        with torch.no_grad(): # no grad when test and predict
            for data in test_loader:
                images,labels = data

                outputs = self.net(images)
                predicts = torch.argmax(outputs,1)
                total += labels.size(0)
                correct += (predicts == labels).sum().item()
        
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def mnist_load_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.],[1.])]) #转化成tensor并归一化
    trainset = torchvision.datasets.MNIST(root = './data',train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size = 32,shuffle = True,num_workers = 2)
    testset = torchvision.datasets.MNIST(root = './data',train=False,download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size = 32,shuffle = True,num_workers = 2)
    return trainloader,testloader




#神经网络的搭建类
class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        #形状为28*28，512层的输入层
        self.fc1 = torch.nn.Linear(28*28,512)
        #512X512隐藏层
        self.fc2 = torch.nn.Linear(512,512)
        #512X10输出层
        self.fc3 = torch.nn.Linear(512,10)
    
    def forward(self,x):
        #PyTorch中的张量操作是动态的，它们会根据张量的当前形状自动调整操作的维度。因此，当使用 .view() 函数时，PyTorch会自动根据张量的形状
        # 和维度进行计算，以生成正确的输出形状。
        #在这个特定的例子中，由于输入张量的形状是 (N, *)，其中 * 表示其他维度，.view(-1, 28*28) 将 N 这个维度保留下来，并将其他维度展平为
        #  1，从而得到了一个新的形状为 (N, 28*28) 的张量。这种用法在处理多维数据时非常常见，因为它可以将高维数据转换为低维数据，或者将不同
        # 尺寸的数据进行对齐。
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x
        



#主函数
if __name__ == '__main__' :
    #神经网络的搭建，初始化
    net = MnistNet()
    #网络的损失函数、优化器的初始化
    model = Model(net,'CROSS_ENTROPY','ADAM')
    #mnist数据集的获取
    train_loader,test_loader = mnist_load_data()
    #神经网络的训练
    model.train(train_loader)
    #神经网络的推理
    model.evaluate(test_loader)