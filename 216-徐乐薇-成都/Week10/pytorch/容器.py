import torch.nn as nn

# 方法1
model = nn.Sequential() # Sequential是一个有序的容器，神经网络模块将按照在传入Sequential的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
model.add_module('fc1', nn.Linear(3,4)) # add_module方法可以给每个子模块指定一个名字，以后我们可以根据名字取出子模块, nn.Linear是一个全连接层, 3为输入特征维度, 4为输出特征维度
model.add_module('fc2', nn.Linear(4,2)) # 4为输入特征维度, 2为输出特征维度
model.add_module('output', nn.Softmax(2)) # Softmax层将输出转化成一个概率分布. 2表示dim, 即dim=2, 表示对第2维进行Softmax操作

# 方法2
model2 = nn.Sequential(                 # Sequential的另一种写法
          nn.Conv2d(1,20,5),            # 1表示输入通道数, 20表示输出通道数, 5表示卷积核大小
          nn.ReLU(),                    # 激活函数
          nn.Conv2d(20,64,5),           # 20表示输入通道数, 64表示输出通道数, 5表示卷积核大小
          nn.ReLU()                     # 激活函数
        )
# 方法3        
model3 = nn.ModuleList([nn.Linear(3,4), nn.ReLU(), nn.Linear(4,2)]) # ModuleList可以让模块以追加的方式添加到网络中, ModuleList不同于一般的Python的list, 加入到ModuleList里面的所有模块的参数会被自动添加到整个网络中
