#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date 2023/7/5
@author: 81-yuhaiyang

"""
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from utils import ImgPreview


def gen_data():
    # 数据是固定的， tran 有 60000个  test 10000 个
    (train_img, train_l), (test_img, test_l) = mnist.load_data()
    train_img = train_img.reshape((60000, 28 * 28))
    train_img = train_img.astype('float32') / 255
    # one hot
    train_l = to_categorical(train_l)

    test_img = test_img.reshape((10000, 28 * 28))
    test_img = test_img.astype('float32') / 255
    test_l = to_categorical(test_l)

    print("start gen data", type(test_img), type(test_l))
    return (train_img, train_l), (test_img, test_l)


def gen_predict_data(count: int):
    _, (ori_images, ori_labels) = mnist.load_data()
    pre_images = ori_images.reshape((10000, 28 * 28))
    pre_images = pre_images.astype('float32') / 255

    r, _ = pre_images.shape
    random_index = np.random.choice(r, count)
    return pre_images[random_index], ori_images[random_index], ori_labels[random_index]


def design_model():
    print("design_model")
    m = models.Sequential()
    m.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
    m.add(layers.Dense(1024, activation="selu", input_shape=(28 * 28,)))
    m.add(layers.Dense(10, activation="softmax"))
    m.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return m


if __name__ == '__main__':
    print("Hi Keras")
    predict_count = 6

    (train_images, train_labels), (test_images, test_labels) = gen_data()
    (predict_img, predict_img_ori, predict_label) = gen_predict_data(predict_count)

    # Design、Train
    model = design_model()
    model.fit(train_images, train_labels, epochs=5, batch_size=128)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    res = model.predict(predict_img)

    # Preview
    preview = ImgPreview(20, 6, 1, predict_count)
    for i in range(predict_count):
        preview.add(predict_img_ori[i], grey=True)
    preview.show()

    print(res)

    for i in range(res.shape[0]):
        predict_res = np.argmax(res[i])
        predict_act = predict_label[i]
        print(f"PREDICT:{predict_res}, ACTUAL:{predict_act}, RES:{predict_res == predict_act}")
