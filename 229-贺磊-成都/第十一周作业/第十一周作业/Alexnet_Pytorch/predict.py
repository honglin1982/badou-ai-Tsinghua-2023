# -*- coding: utf-8 -*-
# File  : predict.py
# Author: HeLei
# Date  : 2023/8/8

import numpy as np
import cv2
import torch
from model.AlexNet import AlexNet
from torchvision import transforms
from PIL import Image

# 注意在gpu上面训练的话，推理的时候需要移植到cpu上或者使用gpu推理。
# 定义加载图片的方式
transformed = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

if __name__ == '__main__':
    model = AlexNet()
    mymodel = torch.load(r"F:\AI_Learn\第十一周\VGG_Pytorch\nets\cifar10_model_21.pt")

    # img = cv2.imread("dog.png")
    img = Image.open("dog.png")
    # img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_RGB = img.convert("RGB")
    img = transformed(img_RGB)
    img = torch.reshape(img, (1, 3, 32, 32))

    mymodel.eval()
    with torch.no_grad():
        output = mymodel(img)
        x = output.argmax(1)
        print("x==",x)
        if x == 5:
            print("该图片可能为狗")
        else:
            print("预测出错！")
