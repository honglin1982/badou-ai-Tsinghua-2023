[1]
'''
将训练数据和检测数据加载到内存中(第一次运行需要下载数据，会比较慢):
train_images是用于训练系统的手写数字图片;
train_labels是用于标注图片的信息;
test_images是用于检测系统训练效果的图片；
test_labels是test_images图片对应的数字标签。
'''
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape:', train_images.shape)
print('train_labels.shape:', train_labels.shape)
print('test_images.shape:', test_images.shape)
print('test_labels.shape:', test_labels.shape)

[2]
'''
把用于测试的第一张图片打印出来看看
'''
import matplotlib.pyplot as plt
digit = train_images[0]
plt.imshow(digit, cmap=plt.cm.binary) #cmap=plt.cm.binary表示以灰度图的形式显示图片
plt.show()


[3]
'''
使用tensorflow.Keras搭建一个有效识别图案的神经网络，
1.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
2.models.Sequential():表示把每一个数据处理层串联起来.
3.layers.Dense(…):构造一个数据处理层。
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
'''
from tensorflow.keras import models
from tensorflow.keras import layers
network = models.Sequential() #创建一个神经网络,网络中每一个数据处理层都是一个对象
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,))) #添加一个数据处理层,relu:激活函数
network.add(layers.Dense(10, activation='softmax')) #添加一个数据处理层,softmax分类器
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) #编译网络,optimizer:优化器,loss:损失函数,metrics:评估指标

[4]
'''
在把数据输入到网络模型之前，把数据做归一化处理:
1.reshape(60000, 28*28）:train_images数组原来含有60000个元素，每个元素是一个28行，28列的二维数组，
现在把每个二维数组转变为一个含有28*28个元素的一维数组.
2.由于数字图案是一个灰度图，图片中每个像素点值的大小范围在0到255之间.
3.train_images.astype(“float32”)/255 把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值。
'''
train_images = train_images.reshape(60000, 28*28) #把每个二维数组转变为一个含有28*28个元素的一维数组
train_images = train_images.astype('float32')/255 #把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值

test_images = test_images.reshape(10000, 28*28) #把每个二维数组转变为一个含有28*28个元素的一维数组
test_images = test_images.astype('float32')/255 #把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值

'''
把图片对应的标记也做一个更改：
目前所有图片的数字图案对应的是0到9。
例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7。
我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0。
例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0] ---one hot
'''
from tensorflow.keras.utils import to_categorical #把标记转变为one hot编码
print("one hot前:", test_labels[0]) #打印出test_labels[0]的值
train_labels = to_categorical(train_labels) #把标记转变为one hot编码
test_labels = to_categorical(test_labels) #把标记转变为one hot编码
print("one hot后:", test_labels[0])

[5]
'''
把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次,每次循环都会把所有的图片都计算一遍,循环多次是为了提高识别的准确率.
'''
network.fit(train_images, train_labels, epochs=5, batch_size=128) #把数据输入网络进行训练

[6]
'''
测试数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
'''
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1) #测试数据输入，检验网络学习后的图片识别效果
print('test_acc:', test_acc)
print('test_loss:', test_loss)

[7]
'''
输入一张手写数字图片到网络中，看看它的识别效果
'''
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[20]
plt.imshow(digit, cmap=plt.cm.binary) #把手写数字图片显示出来
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images) #输入手写数字图片到网络中，看看它的识别效果.res是一个含有10个元素的数组，每个元素的值都是0到1之间的浮点数

for i in range(res[20].shape[0]): #res[20]表示res数组中的第20个元素，res[20].shape[0]表示res[20]数组的长度
    if (res[20][i] == 1):
        print("the number for the picture is : ", i)
        plt.imshow(digit, cmap=plt.cm.binary)
        plt.title("Predicted number: {}".format(i))
        plt.show()
        break