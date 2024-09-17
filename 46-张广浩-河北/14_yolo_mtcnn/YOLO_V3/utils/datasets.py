import argparse
import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable


from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

#p2d = (1, 2, 3, 4)  # (左边填充数， 右边填充数， 上边填充数， 下边填充数)
#t2 = F.pad(t4d, p2d, 'constant', 2)
#图像填充
from utils.augmentations import horisontal_flip
from utils.parse_config import parse_data_config


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding (375,500)->(500,500)进行填充
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    #F.interpolate 功能：利用插值方法，对输入的张量数组进行上\下采样操作
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        """
        :param list_path: data/coco/5k.txt 标注路径
        :param img_size: 图像尺寸
        :param augment: 是否图像增强
        :param multiscale: 是否多尺度检测
        :param normalized_labels:
        """
        self.img_name = []
        self.label_files = []
        self.img_files = []
        with open(list_path, "r") as file:
            self.img_filess = file.readlines()
        #数据集和项目文件不在同一个文件夹下，变'D:/python/data/COCO/val2014/COCO_val2014_000000000164.txt\n' 找不到
        #所以把数据集标注文件也放在了data/COCO下
        #图像路径：D:/python/data/COCO/images/val2014/COCO_val2014_000000000164.jpg
        #标注路径：D:/python/data/COCO/lables/val2014/COCO_val2014_000000000164.txt
        #D:\python\data\COCO\labels
        # self.label_files = [
        #     #路径变化，
        #     # images->labels,
        #     # .png /.jpg ->.txt
        #     #/images/val2014/COCO_val2014_000000002261.jpg
        #     #/labels/val2014/COCO_val2014_000000002261.txt
        #     path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
        #     for path in self.img_files
        # ]
        self.test_or_train = False
        train_or_test = list_path.split("/")[-1].split(".")[0]
        if train_or_test == "5k":
            self.label_root = r'D:\python\data\COCO\labels\val2014'
            self.image_root = r'D:\python\data\COCO\images\val2014'
            self.test_or_train = True
        else:
            self.label_root = r'D:\python\data\COCO\labels\train2014'
            self.image_root = r'D:\python\data\COCO\images\train2014'
            self.test_or_train = False
        #self.label_root = r'D:\python\data\COCO\labels\val2014'
        for path in self.img_filess:
            filename = path.split("/")[-1].split(".")[0]
            self.img_name.append(filename)
            file_class = filename.split("_")[1]
            if self.test_or_train == False and file_class == "val2014":
                img_root_path = r'D:\python\data\COCO\images\val2014'
                lable_root_path = r'D:\python\data\COCO\labels\val2014'
                img_path = os.path.join(img_root_path, filename + ".jpg")
                label_path = os.path.join(lable_root_path, filename + ".txt")
            else:
                img_path = os.path.join(self.image_root, filename + ".jpg")
                label_path = os.path.join(self.label_root,filename+".txt")
            self.label_files.append(label_path)
            self.img_files.append(img_path)

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        # 416-32*3 = 320
        # 416+32*3 = 512
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        #.rstrip() 返回删除 string 字符串末尾的指定字符后生成的新字符串。
        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        #通道数压缩为3
        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------
        #14 0.820783 0.561290 0.218633 0.352700
        #14 0.293458 0.376340 0.159617 0.334840
        #14 0.525983 0.416530 0.166667 0.325540
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            #不只一个目标
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h
            #targets: num_obj , classes , x , y , w ,h 其中x,y,w,h都是比例
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        #horisontal_flip函数作用是将图片左右翻转，来做图像增强
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets
    #这部分代码是将pad后的图像进行resize成
    # torch.Size([1, 3, 320, 320]),
    # torch.Size([1, 3, 384, 384]),
    # torch.Size([1, 3, 480, 480])三种尺度的图像。
    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets图中目标与该图索引对应
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            #在320和512之间随机选择尺寸进行图像裁剪，实现多尺度训练。以32为一个层 ，320 352 384……512
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


if __name__ =="__main__":
    path = r'D:\python\YOLO\YOLO_V3\data\coco\5k.txt'
    path_val = "../data/coco/5k.txt"
    path_train = "../data/coco/trainvalno5k.txt"
    #这个文件在utils文件夹下，相对路径不一样

    # 得到测试图像路径文件 data/coco/5k.txt
    # 这里对于各个图像路径进行一个修改 ，在5k.txt文件中，图像路径改为图像在本地存储路径

    dataset = ListDataset(path_train,416,augment=False, multiscale=False)
    dataloader = DataLoader(dataset,1,shuffle=True)
    for batch, data in enumerate(dataloader):
        path_return,X, Y = data
        # 将数据放在GPU上训练
        X, Y = Variable(X).cuda(), Variable(Y).cuda()
        print(path_return,X.shape,Y.shape)
        if batch == 5:
            break
