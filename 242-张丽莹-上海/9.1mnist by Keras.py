# 用Keras调接口方式搭建简易神经网络实现mnist分类

# 1.加载数据
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ', train_images.shape)
print('train_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels = ', test_labels)
# train_images.shape =  (60000, 28, 28) 含有60000个行和列都是28的数组
# train_labels =  [5 0 4 ... 5 6 8]  第一张手写数字图片的内容是数字5，第二种图片是数字0，以此类推.
# test_images.shape =  (10000, 28, 28)
# test_labels =  [7 2 1 ... 4 5 6]

# 2.打印测试第一张图片
import matplotlib.pylab as plt
digit = test_images[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# 3.用tensorflow.Keras搭建一个有效识别图案的神经网络
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])

# 4. 数据 标签处理
# 训练集测试集 数据 reshape + 归一化处理
train_images = train_images.reshape((60000, 28*28)) # shape变成（60000，784）
train_images = train_images.astype("float32")/255  # 每个像素点的值从范围0-255转变为范围在0-1之间的浮点值
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# 对训练集测试集 标签 进行one-hot处理.如testlabel[0]由 1 变成10元素列表[1,0,0。。。]
from tensorflow.keras.utils import  to_categorical
print("before change:", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

# 5.训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 6.测试数据输入，检验网络学习后的图片识别的损失和准确率
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

# 7.现场数据测试，这里直接用了库里的测试集。也可以自行输入，reshape后传入predict函数
res = network.predict(test_images)
print(res.shape)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1): # 找到数值为1的图片
        print("the number for the picture is:", i)
        break
