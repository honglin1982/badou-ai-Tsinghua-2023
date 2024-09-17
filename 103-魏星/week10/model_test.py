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
from pytorch_mnist import NetWork

tf.compat.v1.disable_eager_execution()

def test_keras_model():
    print("keras mnist begin")
    # 指定模型路径和文件名
    model_path = "./model/keras-mnist.h5"
    # 加载模型
    model = tf.keras.models.load_model(model_path)

    img1 = cv2.imread("./testimg/test1.png",0)
    img1_result = model.predict(img1.reshape(-1, 28, 28, 1))
    print("真实答案是：7，预测数字是：", np.argmax(img1_result))

    img2 = cv2.imread("./testimg/test_2.png",0)
    img2_result = model.predict(img2.reshape(-1, 28, 28, 1))
    print("真实答案是：2，预测数字是：", np.argmax(img2_result))

    print("keras mnist end")

def test_torch_model():
    print("torch mnist begin")
    network = NetWork()
    network.load_state_dict(torch.load("./model/torch-mnist.h5"))

    img1 = cv2.imread("./testimg/test1.png", 0)
    outputs = network.forward(torch.tensor(np.float32(img1.reshape(-1, 1, 28, 28))))
    _, img1_result = torch.max(outputs.data, 1)
    print("真实答案是：7，预测数字是：", img1_result.item())

    img2 = cv2.imread("./testimg/test_2.png", 0)
    outputs = network.forward(torch.tensor(np.float32(img2.reshape(-1, 1, 28, 28))))
    _, img2_result = torch.max(outputs.data, 1)
    print("真实答案是：2，预测数字是：", img2_result.item())

    print("torch mnist end")

if __name__=='__main__':
    # test_keras_model()
    # test_torch_model()
    arr = [7]
    one_hots = np.eye(10)[arr]
    print(one_hots)
    pass