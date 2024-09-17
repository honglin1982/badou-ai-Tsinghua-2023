# from tensorflow import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
import numpy as np

(train_image, train_labels), (test_image, test_labels) = mnist.load_data()
# print(f'train_image: {train_image}\n: {train_image.shape}')
print(f'train_image: {train_image.shape}')
print(f'train_labels: {train_labels}: {len(train_labels)}')
# print(f'train_image: {test_image}\n: {test_image.shape}')
print(f'test_image: {test_image.shape}')
print(f'test_labels: {test_labels}: {len(test_labels)}')


# 数据预处理
train_image = train_image.reshape((60000, 28 * 28))  # 将二维数组装转化为一维
train_image1 = train_image.astype('float32') / 255
test_image = test_image.reshape((10000, 28 * 28))  # 将二维数组装转化为一维
test_image1 = test_image.astype('float32') / 255

# one hot
train_labels1 = to_categorical(train_labels)
test_labels1 = to_categorical(test_labels)
'''
使用tensorflow.Keras搭建一个有效识别图案的神经网络，
1.models.Sequential():表示把每一个数据处理层串联起来.
2.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
3.layers.Dense(…):构造一个数据处理层。
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
'''
# 创建模型
network = models.Sequential()  # 串联
network.add(layers.Dense(units=512, activation='relu', input_shape=(28*28,)))  # 隐藏层节点
network.add(layers.Dense(units=10, activation='softmax'))  # 输出层节点
'''
sgd = SGD(lr=0.001) #学习率lr
# 激活神经网络
model.compile(
        optimizer = sgd,                 # 加速神经网络
        loss = 'categorical_crossentropy',   # 损失函数
        metrics = ['accuracy'],               # 计算误差或准确率
        )
'''
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 使用训练集 并且训练5轮 batch_size为128
network.fit(train_image1, train_labels1, batch_size=128, epochs=5)
# 使用测试集评估模型
loss, acc = network.evaluate(test_image1, test_labels1, verbose=1)
print(f'---loss: {loss}')
print(f'---acc: {acc}')
# models.save_model(network, filepath='./epic_num_reader.model')
'''
model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict([x_test])
print(predictions)

'''
# 自己读入一张图来推理
# random_choice_image = cv2.imread('./1341689155350_.png')
# random_choice_image = cv2.imread('./img7.png')
# random_choice_image = cv2.imread('./img5.png')
# random_choice_image = cv2.cvtColor(random_choice_image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("999", random_choice_image)
# cv2.waitKey(1000)
# random_choice_image = cv2.resize(random_choice_image, dsize=(28, 28)).reshape((1, 28*28))
# # random_choice_image = random_choice_image.astype('float32') / 255
# print(random_choice_image)
# res = network.predict(random_choice_image)
# print("**" * 30)
# print(res)
# print(res.shape)
# print(res[0])
# print(np.argmax(res.flatten()))

(train_x, train_y), (test_x, test_y) = mnist.load_data()
random_choice_image = test_x[0]
plt.imshow(random_choice_image, cmap=plt.cm.binary)
plt.show()
res = network.predict(test_image)
for i in range(res[0].shape[0]):
    if res[0][i] == 1:
        print("the number for the picture is : ", i)
        break






