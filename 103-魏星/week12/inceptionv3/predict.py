# _*_ coding : utf-8 _*_
# @Time : 2023/7/24 13:37
# @Author : weixing
# @FileName : predict
# @Project : cv

import numpy as np
import cv2
import torch
import os
import json

from InceptionV3 import InceptionV3

labels_cn = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
Model_File_Url = "./torch-inceptionV3.h5"  # 模型保存路径

if __name__ == "__main__":
    print("torch inceptionV3 begin")
    model = InceptionV3(num_classes=10)
    model.load_state_dict(torch.load(Model_File_Url))

    img1 = cv2.imread("./Test.jpg")
    img1 = cv2.resize(img1, (299, 299))
    outputs = model.forward(torch.tensor(np.float32(img1.reshape(-1, 3, 299, 299))))
    _, img1_result = torch.max(outputs.data, 1)

    # read class_indict
    json_path = './cifar10_class.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    print("模型预测结果是：", class_indict[str(img1_result.item())])
    # print("模型预测结果是：", labels_cn[int(img1_result.item())])

    print("torch inceptionV3 end")