from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import numpy as np

'''
用keras实现简单神经网络
从零开始实现神经网络
'''


(x_train, y_train), (x_test, y_test_origin) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0 * 0.99 + 0.01
y_train = to_categorical(y_train)
x_test = x_test.reshape((x_test.shape[0], -1)) / 255.0 * 0.99 + 0.01
y_test = to_categorical(y_test_origin)


sequential = models.Sequential()
sequential.add(layers.Dense(units=512, activation="sigmoid"))
sequential.add(layers.Dense(units=10, activation="softmax"))
sequential.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=losses.categorical_crossentropy)

sequential.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)

predictions = sequential.predict(x_test, batch_size=64)
correct_num = 0
for index in range(predictions.shape[0]):
    predict = np.argmax(predictions[index])
    true_value = y_test_origin[index]
    if predict == true_value:
        correct_num += 1
print("预测%d个，正确%d个，正确率：%.4f" % (predictions.shape[0], correct_num, correct_num / predictions.shape[0]))