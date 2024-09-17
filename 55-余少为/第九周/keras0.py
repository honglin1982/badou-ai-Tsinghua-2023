from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np


# 加载训练集和测试集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_labels[:5], train_labels.shape)

# 数据归一化处理
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# 图片标签做one-hot编码处理
print('before change:', test_labels[0])
train_labels = to_categorical(train_labels)
test_labels_1 = to_categorical(test_labels)
print('after change:', test_labels[0])


# 搭建一个简单的神经网络
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))    # 28*28 => 512
# network.add(layers.Dense(512, input_shape=(28*28,)))    # 去掉relu之后，模型测试正确率也能达到0.917
network.add(layers.Dense(10, activation='softmax'))     # 512 => 10

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 用训练数据进行拟合
network.fit(train_images, train_labels, batch_size=128, epochs=5)

# 用测试数据评估模型的训练效果
test_loss, test_acc = network.evaluate(test_images, test_labels_1, verbose=0)
print(test_loss)
print('test_acc:', test_acc)

# 取前十张测试图片，看看预测结果
res = network.predict(test_images[:10])
print('前10张手写字的预测值为：', np.argmax(res, axis=1))
print('前10张手写字的真实值为：', test_labels[:10])
