# _*_ coding : utf-8 _*_
# @Time : 2023/7/24 8:53
# @Author : weixing
# @FileName : cifar_10_dataset
# @Project : cv

'''
CIFAR-10 是由 Hinton 的学生 Alex Krizhevsky 和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。
一共包含 10 个类别的 RGB 彩色图 片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。
图片的尺寸为 32×32 ，数据集中一共有 50000 张训练圄片和 10000 张测试图片。
与 MNIST 数据集中目比， CIFAR-10 具有以下不同点：
• CIFAR-10 是 3 通道的彩色 RGB 图像，而 MNIST 是灰度图像。
• CIFAR-10 的图片尺寸为 32×32， 而 MNIST 的图片尺寸为 28×28，比 MNIST 稍大。
• 相比于手写字符， CIFAR-10 含有的是现实世界中真实的物体，不仅噪声很大，而且物体的比例、 特征都不尽相同，这为识别带来很大困难。
    直接的线性模型如 Softmax 在 CIFAR-10 上表现得很差。
'''

from tensorflow import keras
# import loadLoaclCifar10 as cifar10

# 加载数据集
cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# train_images, train_labels, x_val, y_val, test_images, test_labels = cifar10.get_CIFAR10_data()
print(test_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# 对图像进行预处理
train_images = train_images / 255.0
test_images = test_images / 255.0

# 构建模型
model = keras.Sequential([
    keras.layers.Conv2D(28, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(26, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(416, activation='relu'),
    keras.layers.Dense(208, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_images.reshape(-1, 32, 32, 3), train_labels, epochs=20, batch_size=150)

# 评估模型
test_loss, test_acc = model.evaluate(test_images.reshape(-1, 32, 32, 3), test_labels)
print('Test accuracy:', test_acc)

# 预测结果
predictions = model.predict(test_images.reshape(-1, 32, 32, 3))
# print(predictions)

# 保存模型
model_path = "./model/keras_cifar10.h5"
model.save(model_path)

