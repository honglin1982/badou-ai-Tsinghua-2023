#! /usr/bin/python
# -*- coding:utf-8 -*-
import matplotlib
from tensorflow.keras.datasets import mnist

(train_images, tarin_labels), (test_images, test_labels) = mnist.load_data()
print("train_images.shape:", train_images.shape)
print("train_labels:", tarin_labels)
print("test_images.shape:", test_images.shape)
print("test_labels:", test_labels)
'''
从tensorflow中加载测试数据和训练数据
分别为训练数据和测试数据
内部又分为测试图像和标签
'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

digits = test_images[0]

import matplotlib.pyplot as plt
plt.imshow(digits, cmap=plt.cm.binary)
input('按下回车键继续')
plt.show(block=False)
'''
打印出训练数据的第一张图片
'''

from tensorflow.keras import models
from tensorflow.keras import layers

'''
从tensorflow中引入模型和层
'''

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
'''
实例化一个模型，然后构造一个三层神经网络，输入层为28*28，隐藏层为512，输出层为10，
因为是连续的模型，所以内部会直接全相连，且下层的输入模式是上层的输出
layers：表示神经网络中的一个数据处理层（dense为全联接层）
models.Sequential表示将每一个数据处理层连接起来
layers.Dense()表示构造一个数据处理层
'''

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
'''
将数据调整为需要的格式
在数据输入模型之前，要进行归一化处理
'''

from tensorflow.keras.utils import to_categorical

print("before change:", test_labels[0])
test_labels = to_categorical(test_labels)
train_labels = to_categorical(tarin_labels)
print("after change:", test_labels[0])
'''
这里引入to_categorical来将标签转换为独热编码
'''

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print(test_acc)

# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# digit = test_images[1]
# plt.imshow(digit,cmap=plt.cm.binary)
# plt.show()
# test_images = test_images.reshape((10000, 28 * 28))
# res = network.predict(test_images)

# for i in range(res[1].shape[0]):
#     if (res[1][i] == 1):
#         print("the number is:", i)
#         break


my_digits = cv2.imread("handwrite2.jpg")
my_digits = cv2.cvtColor(my_digits,cv2.COLOR_BGR2GRAY)
my_digits = cv2.resize(my_digits,(28,28))
# my_digits = my_digits / 255.0
plt.imshow(my_digits,cmap=plt.cm.binary)
plt.show()
my_digits = my_digits.reshape(1,(28*28))
res = network.predict(my_digits)

for i in range(res[0].shape[0]):
    if (res[0][i] == 1):
        print("the number is:", i)
        break


for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number is:", i)
        break

input("按下回车键继续")
plt.close()
