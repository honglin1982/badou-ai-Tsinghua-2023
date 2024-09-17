import torch

class Linear(torch.nn.Module):                                          # 继承 torch 的 Module，pytorch 的神经网络模块化程度很高，只需要定义好 forward 函数，backward函数会在使用autograd时自动定义
    def __init__(self, in_features, out_features, bias=True):           # 定义构造函数, 传入参数in_features为输入特征数, out_features为输出特征数, bias为是否使用偏置
        super(Linear, self).__init__()                                  # 调用父类的构造函数，下面继承了父类的属性，torch.nn.Module具有持久化特性，即在类中定义的成员在反复调用中始终存在
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features)) # 使用 Parameter 定义参数，具有持久化特性
        if bias:                                                            # 如果有偏置，定义并使用 Parameter 定义
            self.bias = torch.nn.Parameter(torch.randn(out_features))   # 定义并使用 Parameter 定义

    def forward(self, x):                                               # 定义前向传播函数，接受输入并返回输出
        x = x.mm(self.weight)                                           # torch.mm是矩阵乘法，x为输入，self.weight为在构造函数中定义的模型参数
        if self.bias:                                                   # 如果有偏置项，计算后加上偏置
            x = x + self.bias.expand_as(x)                              # expand_as(x)将bias的维度扩展为x的维度
        return x                                                        # 返回结果

if __name__ == '__main__':
    # train for mnist                                                        # 使用偏置
    net = Linear(3,2)                                                        # 输入特征数为3，输出特征数为2的线性层,比如输入为[0.1,0.2,0.3],输出为[0.1,0.2]
    x = net.forward                                                          # 输入给网络，输出结果
    print('11',x)                                                            # 输出随机数
