# -*- coding: utf-8 -*-
# File  : My_keras.py
# Author: HeLei
# Date  : 2023/7/2
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from matplotlib import pyplot as plt
import numpy as np

# No1:获取数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ', train_images.shape)  # 获取训练数据集的维度
print('tran_labels = ', train_labels)  # 训练数据集的标签
print('test_images.shape = ', test_images.shape)  # 测试数据集的维度
print('test_labels', test_labels)  # 测试数据集的的标签

digit = test_images[0]  # 画图看一看
plt.imshow(digit, cmap="gray")
plt.show()

# No2：构建神经网络模型
network = models.Sequential()
network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))  # 中间隐藏层
network.add(layers.Dense(10, activation="softmax"))  # 输出层

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])

# No3：归一化处理
train_images = train_images.reshape((60000, 28 * 28))  # 将28*28的二维数据转成一维数组
train_images = train_images.astype('float32') / 255  # 归一化到0与1之间

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 使用one-hot编码对标签结果进行处理
print("before change:", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

# No4：训练
network.fit(train_images, train_labels, epochs=10, batch_size=128)

# No5：评估
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print("test_loss", test_loss)
print("test_acc", test_acc)

# No5：推理
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# digit = test_images[1]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()
# test_images = test_images.reshape((10000, 28 * 28))
# res = network.predict(test_images)
#
# for i in range(res[1].shape[0]):
#     if res[1][i] == 1:
#         print("the number for the picture is : ", i)
#         break

# 用自己的图片推理
img = cv2.imread("my_own_4.png", 0)  # 读图
new_img = cv2.resize(img, (28, 28))  # 裁剪到28*28

plt.imshow(new_img, cmap=plt.cm.binary)
plt.show()
new_img = new_img.astype('float32') / 255
image = new_img.reshape(1, 28 * 28)

result = network.predict(image)
print("the number for the picture is : ", np.argmax(result, axis=1)[0])
