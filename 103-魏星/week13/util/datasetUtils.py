# _*_ coding : utf-8 _*_
# @Time : 2023/8/8 14:29
# @Author : weixing
# @FileName : datasetUtils
# @Project : cv


# 11 数据集划分
import os
import random
import xml.etree.ElementTree as ET

random.seed(1)


def data_ide(xmlfilepath, saveBasePath, trainval_perscent=1, train_persent=1, num=None):
    temp_xml = os.listdir(xmlfilepath)  # 返回指定路径下的文件和文件夹列表
    total_xml = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)

    if num == None:
        num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_perscent)
    tr = int(tv * train_persent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')
    for i in list:
        if i < len(total_xml):
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)
        else:
            print("error: list out of range, please decrease 'num'")

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


def convert_annotation(xmlfilepath, image_id, list_file, classes):
    in_file = open(os.path.join(xmlfilepath, str(image_id) + '.xml'), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    # print(image_id)
    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
            # classes.append(cls)
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)),
             int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)),
             int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + "," + str(cls_id))
        # print(classes)


def voc_annotation(sets, classes, imagePath, BasePath, xmlfilepath):
    # wd = os.getcwd()  # 返回当前工作目录
    for year, image_set in sets:
        image_ids = open(os.path.join(BasePath, str(image_set) + '.txt'), encoding='utf-8').read().strip().split()
        list_file = open(os.path.join(BasePath, '{}_{}.txt'.format(year, image_set)), 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file.write(os.path.join(imagePath, str(image_id) + '.jpg'))
            convert_annotation(xmlfilepath, image_id, list_file, classes)
            list_file.write('\n')
        list_file.close()
