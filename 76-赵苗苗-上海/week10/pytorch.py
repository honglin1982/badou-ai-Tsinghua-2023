#使用pytorch框架训练和评估mnist手写数字识别模型
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
#1、定义Model类（封装了神经网络模型，损失函数和优化器的初始化、创建、训练和评估方法）
class Model:
    #net——神经网络模型，cost——损失函数类型，optimist——优化器类型
    def __init__(self,net,cost,optimist):
        #初始化模型，损失函数和优化器
        self.net=net
        self.cost=self.create_cost(cost)
        self.optimizer=self.create_optimizer(optimist)
    def create_cost(self,cost):
        #创建损失函数对象
        support_cost={
            'CROSS_ENTROPY':nn.CrossEntropyLoss(),
            'MSE':nn.MSELoss()
        }
        return support_cost[cost]
    def create_optimizer(self,optimist,**rests):
        #创建优化器对象
        support_optim={
            'SGD':optim.SGD(self.net.parameters(),lr=0.1,**rests),
            'ADAM':optim.Adam(self.net.parameters(),lr=0.01,**rests),
            'RMSP':optim.RMSprop(self.net.parameters(),lr=0.001,**rests)
        }
        return support_optim[optimist]
    def train(self,train_loader,epoches=3):
        #训练模型
        self.net.train()  #设置模式为训练模式
        for epoch in range(epoches):
            running_loss=0.0
            for i,data in enumerate(train_loader,0):
                inputs,labels=data  #从训练数据中获取输入数据和对应的标签
                self.optimizer.zero_grad()  #将优化器的梯度缓冲区清零（保证不受之前迭代信息的影响）
                
                outputs=self.net(inputs)
                loss=self.cost(outputs,labels)
                loss.backward()
                self.optimizer.step()

                running_loss+=loss.item()
                if i % 100==0:
                    print('[epoch%d,%.2f%%] loss:%.3f'%
                          (epoch+1,(i+1)*1./len(train_loader),running_loss/100))
                    running_loss=0.0
        print('Finished Training')

    def evaluate(self,test_loader):
        #评估模型
        self.net.eval()    #设置模式为评估模式
        print('Evaluating...')
        correct=0
        total=0
        with torch.no_grad():   #表示在评估和预测过程中不需要计算梯度
            for data in test_loader:
                images,labels=data
                outputs=self.net(images)
                predicted=torch.argmax(outputs,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()
        print('Accuracy of the network on the test images:%d %%'%(100*correct/total))
        
#2、定义MnistNet模型类
class MnistNet(torch.nn.Module):
    def __init__(self):
        #定义神经网络结构
        super(MnistNet,self).__init__()   #调用父类torch.的构造函数nn.Moudle的构造函数
        self.fc1=torch.nn.Linear(28*28,512)
        self.fc2=torch.nn.Linear(512,512)
        self.fc3=torch.nn.Linear(512,10)    #创建三个全连接层

    def forward(self,x):
        #定义前向传播过程
        x=x.view(-1,28*28)   #将图像变为一维
        x=F.relu(self.fc1(x)) #第一层
        x=F.relu(self.fc2(x)) #第二层
        x=F.softmax(self.fc3(x),dim=1) #softmax归一化转化为概率分布
        return x

#3、定义mnist_load_data函数加载mnist数据集
def mnist_load_data():
    #数据预处理和加载
    #将图像转换为张量，对图像数据进行标准化处理
    transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,],[1,])]
    )
    #创建一个训练集对象trainset
    trainset=torchvision.datasets.MNIST(root='./data',train=True,
                                        download=True,transform=transform)
    #创建一个训练数据加载器trainloader
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=32,
                                           shuffle=True,num_workers=2)
    #创建一个测试集对象testset
    testset=torchvision.datasets.MNIST(root='./data',train=False,
                                        download=True,transform=transform)
    #创建一个测试数据加载器testloader
    testloader=torch.utils.data.DataLoader(testset,batch_size=32,
                                           shuffle=True,num_workers=2)
    return trainloader,testloader

#4、在主程序中进行训练和评估模型
if __name__=='__main__':
    #创建模型实例和Model实例
    net=MnistNet()
    model=Model(net,'CROSS_ENTROPY','RMSP')
    #加载mnist数据集
    train_loader,test_loader=mnist_load_data()
    #训练模型
    model.train(train_loader)
    #评估模型
    model.evaluate(test_loader)
