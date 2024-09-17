from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# 加载测试数据集
(train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data()
print("train_img shape:", train_imgs.shape)
print("train_labels shape:", train_labels.shape)
print("test_img shape:", test_imgs.shape)
print("test_labels shape:", test_labels.shape)

# 初始化模型
network = models.Sequential()
# 隐藏层512个神经元
network.add(layers.Dense(512, activation='sigmoid', input_shape=(28*28,)))
# 输出层 输出1~10的概率
network.add(layers.Dense(10, activation='softmax'))
# loss MSE
network.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])

# 把二维数组 转换为一维 做归一化
cnv_train_imgs = train_imgs.reshape((60000, 28 * 28))
cnv_train_imgs = cnv_train_imgs.astype('float32') / 255

cnv_test_imgs = test_imgs.reshape((10000, 28 * 28))
cnv_test_imgs = cnv_test_imgs.astype('float32') / 255

# 结果的标签做one hot 方便获取结果
print("train_labels:", train_labels)
train_labels = to_categorical(train_labels)
print("convert train_labels:", train_labels)
test_labels = to_categorical(test_labels)

# 进行训练
network.fit(cnv_train_imgs, train_labels, epochs=5, batch_size=200)

# 测试 verbose: 0 或 1。日志显示模式。 0 = 安静模式，1 = 进度条。
test_loss, test_acc = network.evaluate(cnv_test_imgs, test_labels)
print("test_loss:", test_loss)
print("test_acc:", test_acc)

test_img = test_imgs[1]
plt.imshow(test_img, cmap=plt.cm.gray)
plt.show()

res = network.predict(cnv_test_imgs)

print("推理结果：", res[1])

print("the number is: ", np.argmax(res[1]))






