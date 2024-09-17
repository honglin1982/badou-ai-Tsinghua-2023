# _*_ coding : utf-8 _*_
# @Time : 2023/7/18 17:16
# @Author : weixing
# @FileName : keras-minist
# @Project : cv

'''
1、数据集准备：准备一个手写数字数据集，可以使用MNIST数据集，该数据集包含了60000张28x28的灰度图像作为训练集和10000张测试图像。
2、数据预处理：对图像进行预处理，例如将图像转换为灰度图像、进行缩放、归一化等操作。
3、构建模型：构建一个卷积神经网络（CNN）模型，用于对手写数字进行分类。使用TensorFlow中的卷积层、池化层、全连接层等模块来构建模型。
4、训练模型：使用训练集对模型进行训练，并使用测试集进行验证。使用TensorFlow中的优化器、损失函数等模块来训练模型。
5、预测结果：使用训练好的模型对新的手写数字图像进行预测，并输出预测结果。
'''

from tensorflow import keras

# 加载数据集
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 对图像进行预处理
train_images = train_images / 255.0
test_images = test_images / 255.0

# 构建模型
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=5, batch_size=32)

# 评估模型
test_loss, test_acc = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
print('Test accuracy:', test_acc)

# 预测结果
predictions = model.predict(test_images.reshape(-1, 28, 28, 1))
# print(predictions)

# 保存模型
model_path = "./model/keras-mnist.h5"
model.save(model_path)
