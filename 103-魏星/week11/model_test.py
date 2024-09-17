# _*_ coding : utf-8 _*_
# @Time : 2023/7/19 16:30
# @Author : weixing
# @FileName : model_test
# @Project : cv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import cv2
import numpy as np
import torch

tf.compat.v1.disable_eager_execution()

labels_cn = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def test_keras_model():
    print("keras cifar10 begin")
    # 指定模型路径和文件名
    model_path = "./model/keras_cifar10.h5"
    # 加载模型
    model = tf.keras.models.load_model(model_path)

    img1 = cv2.imread("./cifar_10_data/test/automobile/261.jpg")
    img1 = img1 / 255.0
    img1_result = model.predict(img1.reshape(-1, 32, 32, 3))
    print("模型预测结果是：", labels_cn[np.argmax(img1_result)])

    img2 = cv2.imread("./cifar_10_data/test/horse/85.jpg")
    img2 = img2 / 255
    img2_result = model.predict(img2.reshape(-1, 32, 32, 3))
    print("模型预测结果是：", labels_cn[np.argmax(img2_result)])

    print("keras cifar10 end")



if __name__=='__main__':
    test_keras_model()
    pass