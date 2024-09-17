# -*- coding: utf-8 -*-
# @Time    : 2023/7/21 20:59
# @Author  : zgh
# @FileName: Mydataset.py
# @Software: PyCharm
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms


class My_Dataset(Dataset):
    def __init__(self,filename,transform=None):
        self.filename = filename   # 文件路径
        self.transform = transform # 是否对图片进行变化
        self.image_name,self.label_image = self.operate_file()

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self,idx):
        # 由路径打开图片
        image = Image.open(self.image_name[idx])
        # 下采样： 因为图片大小不同，需要下采样为227*227
        trans = transforms.RandomResizedCrop(227)
        image = trans(image)
        # 获取标签值
        label = self.label_image[idx]
        # 是否需要处理
        if self.transform:
            image = self.transform(image)
        # 转为tensor对象
        label = torch.from_numpy(np.array(label))
        return image,label

    def operate_file(self):
        # 获取所有的文件夹路径 '../data/net_train_images'的文件夹
        dir_list = os.listdir(self.filename)
        # 拼凑出图片完整路径 '../data/net_train_images' + '/' + 'xxx.jpg'
        full_path = [self.filename+'/'+name for name in dir_list]
        # 获取里面的图片名字
        name_list = []
        for i,v in enumerate(full_path):
            temp = os.listdir(v)
            temp_list = [v+'/'+j for j in temp]
            name_list.extend(temp_list)
        # 由于一个文件夹的所有标签都是同一个值，而字符值必须转为数字值，因此我们使用数字0-4代替标签值
        label_list = []
        temp_list = np.array([0,1,2,3,4],dtype=np.int64) # 用数字代表不同类别
        # 将标签每个复制200个
        for j in range(5):
            for i in range(200):
                label_list.append(temp_list[j])
        return name_list,label_list

class My_Dataset_test(My_Dataset):
    def operate_file(self):
        # 获取所有的文件夹路径
        dir_list = os.listdir(self.filename)
        full_path = [self.filename+'/'+name for name in dir_list]
        # 获取里面的图片名字
        name_list = []
        for i,v in enumerate(full_path):
            temp = os.listdir(v)
            temp_list = [v+'/'+j for j in temp]
            name_list.extend(temp_list)
        # 将标签每个复制一百个
        label_list = []
        temp_list = np.array([0,1,2,3,4],dtype=np.int64) # 用数字代表不同类别
        for j in range(5):
            for i in range(100): # 只修改了这里
                label_list.append(temp_list[j])
        return name_list,label_list

