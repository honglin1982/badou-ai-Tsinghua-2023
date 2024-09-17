# Week10 框架&CNN

## 框架核心组件

- 张量
- 基于张量的操作（Operation）
- 计算图（Computation Graph）
- 自动微分工具（Automatic Differentiation）
- BLAS、cuBLAS、cuDNN 扩展包

## 主流框架

#### 国外

TensorFlow（Google）、Pytorch（Facebook）、MXNet（Amazon）、CNTK（Microsoft）

#### 国内

MindSpore(华为)、PaddlePaddle（百度）、X-Deep Learning(阿里巴巴)、MACE（小米）

## 框架标准-ONNX （Open Neural Network Exchange）

ONNX是一个表示深度学习模型的开放格式。它使用户可以更轻松地在不同框架之间转移模型。
例如，它允许用户构建一个PyTorch模型，然后使用MXNet运行该模型来进行推理。

ONNX最初由微软和Facebook联合发布，后来亚马逊也加入进来，并发布了V1版本，宣布支持ONNX 的公司还有AMD、ARM、华为、
IBM、英特尔、Qualcomm等。

## TensorFlow

1. 使用图 (graph) 来表示计算任务.
2. 在被称之为 会话 (Session) 的上下文 (context) 中执行图.
3. 使用 tensor 表示数据.
4. 通过 变量 (Variable) 维护状态.
5. 使用 feed 和 fetch 可以为任意的操作(arbitrary operation)赋值或者从其中获取数据

其他：TensorBoard 可视化工具

```
# 生成一个具有写权限的日志文件操作对象，将当前命名空间的计算图写进日志中
writer = tf.summary.FileWriter('logs', tf.get_default_graph())
writer.close()
# 启动tensorboard服务(在命令行启动)
tensorboard --logdir logs
```

## Pytorch

### 优势

1. 简洁:
   PyTorch的设计追求最少的封装，尽量避免重复造轮子。不像 TensorFlow 中充斥着session、graph、operation、
   name_scope、variable、tensor、layer等全新的概念，PyTorch 的设计遵循tensor→variable(autograd)→nn.Module
   三个由低到高的抽象层次，分别代表高维数组(张量)、自动求导(变量)和神经网络(层/模块)，而且这三个 抽象之间联系紧密，可以同时进行修改和操作。
2. 速度:
   PyTorch 的灵活性不以速度为代价，在许多评测中，PyTorch 的速度表现胜过 TensorFlow和Keras 等框架。
3. 易用:
   PyTorch 是所有的框架中面向对象设计的最优雅的一个。PyTorch的面向对象的接口设计来源于Torch，而Torch
   的接口设计以灵活易用而著称，Keras作者最初就是受Torch的启发才开发了Keras。
4. 活跃的社区:
   PyTorch 提供了完整的文档，循序渐进的指南，作者亲自维护的论坛，供用户交流和求教问题。Facebook 人 工智能研究院对 PyTorch
   提供了强力支持。

### 常用工具

1. torch :类似 NumPy 的张量库，支持GPU;
2. torch.autograd :基于 type 的自动区别库，支持 torch 之中的所有可区分张量运行;
3. torch.nn :为最大化灵活性而设计，与 autograd 深度整合的神经网络库;
4. torch.optim:与 torch.nn 一起使用的优化包，包含 SGD、RMSProp、LBFGS、Adam 等标准优化
   方式;
5. torch.multiprocessing: python 多进程并发，进程之间 torch Tensors 的内存共享;
6. torch.utils:数据载入器。具有训练器和其他便利功能;
7. torch.legacy(.nn/.optim) :出于向后兼容性考虑，从 Torch 移植来的 legacy 代码;

## 优化算法

名词：

- original-loss： 整个训练集上的loss
- minibatch-loss： 在一个minibatch上的loss

### BGD 梯度下降算法

为了计算Original-loss的梯度，需要使用训练集全部数据

### SGD

SGD 又称 online 的梯度下降， 每次估计梯度的时候， 只选用一个或几个batch训练样本。
当训练数据过大时，用BGD可能造成内存不够用，那么就可以用SGD了。深度学习使用的训练 集一般都比较大(几十万~几十亿)
。而BGD算法，每走一步(更新模型参数)，为了计算 original-loss上的梯度，就需要遍历整个数据集，这显然效率是很低的。而SGD算法，每次随
机选择一个mini-batch去计算梯度，每走一步只需要遍历一个minibatch(一~几百)的数据。

### Momentum

通常情况我们在训练深度神经网络的时候把数据拆解成一小批一小批地进行训练，这就是我们常用的 mini-batch
SGD训练算法，然而虽然这种算法能够带来很好的训练速度，但是在到达最优点的时候并 不能够总是真正到达最优点，而是在最优点附近徘徊。

另一个缺点就是这种算法需要我们挑选一个合适的学习率，当我们采用小的学习率的时候，会导致网络在训练的时候收敛太慢;当我们采用大的学习率的时候，会导致在训练过程中优化的幅度跳过函数的范围，也就是可能跳过最优点。

我们所希望的仅仅是网络在优化的时候网络的损失函数有一个很好的收敛速度，同时又不至于摆动幅度太大。

# 卷积神经网络（CNN）
卷积网络与我们前面实现的网络不同之处在于，它可以直接接受多维向量，而我们以前实现的网络只
能接收一维向量。

### 卷积
- 卷积操作，其实是把一张大图片分解成好多个小部分，然后依次对这些小部分进行识别。
- 通常我们会把一张图片分解成多个3*3或5*5的”小片“，然后分别识别这些小片段，最后把识别的结果
 集合在一起输出给下一层网络。
- 这种做法在图象识别中很有效。因为它能对不同区域进行识别，假设识别的图片是猫脸，那么我们 就可以把猫脸分解成耳朵，嘴巴，眼睛，胡子等多个部位去各自识别，然后再把各个部分的识别结 果综合起来作为对猫脸的识别。

### 池化层
卷积操作产生了太多的数据，如果没有max pooling对这些数据进行压缩，那么网络的运算量将会 非常巨大，而且数据参数过于冗余就非常容易导致过度拟合。

MaxPull（最大值池化）和 AvgPull（均值池化）