# _*_ coding : utf-8 _*_
# @Time : 2023/7/24 10:23
# @Author : weixing
# @FileName : down&saveCifar10
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

import tensorflow.keras.datasets as datasets
from PIL import Image
import os

(x_train_original, y_train_original), (x_test_original, y_test_original) = datasets.cifar10.load_data()
# print(x_train_original.shape) #(50000, 32, 32, 3)
# print(y_train_original.shape) #(50000, 1)
# print(x_test_original.shape) #(10000, 32, 32, 3)
# print(y_test_original.shape) #(10000, 1)

labels_cn = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

root_path = 'D:\\ProgramData\\project-data\\cv\\11-lesson11\\cifar_10_data'

# 下载并保存训练/测试图片 flag可为train、test
def cifar10_image_save(flag):
    for i in range(x_train_original.shape[0]):
        # print(i,x_train_original[i].shape,y_train_original[i])
        dir_t = root_path+"/"+flag+"/" + labels[int(y_train_original[i])] + "/"
        if not os.path.exists(dir_t):
            os.makedirs(dir_t)

        img = Image.fromarray(x_train_original[i])
        img.save(dir_t + str(i) + '.jpg')

    print(flag, '图片保存完成')

if __name__=='__main__':
    # cifar10_image_save('train')
    # cifar10_image_save('test')
    print('图片保存完成')