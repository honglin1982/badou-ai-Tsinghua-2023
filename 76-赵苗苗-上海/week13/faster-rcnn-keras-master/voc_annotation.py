# 这段代码是用于将PASCAL VOC数据集的标注信息转换为训练所需的文本格式文件
import xml.etree.ElementTree as ET
from os import getcwd

#定义数据集的年份，集合类型以及工作目录路径等变量
sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
wd = getcwd()

#定义类别列表，即数据集中包含的物体类别
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",\
            "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",\
                  "sheep", "sofa", "train", "tvmonitor"]

#定义函数convert_annotation()，将指定图像的标注信息转换为训练所需的文本格式，并写入到指定文件list_file中，在转换过程中，会根据设定的类别列表和特定条件对物体进行过滤和筛选
def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    #如果没有找到object标签，则返回
    if root.find('object')==None:
        return
    #写入图像的路径
    list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
    #遍历每个object标签
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        #如果物体类别不在指定的类别列表中，或者物体的难度为1(难以识别)，则跳过该物体
        if cls not in classes or int(difficult)==1:
            continue
        #获取物体类别在类别列表中的索引号
        cls_id = classes.index(cls)
        #获取bounding box的信息
        xmlbox = obj.find('bndbox')
        #获取bounding box的四个坐标值(xmin,ymin,xmax,ymax)
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        #将上述信息整合成一个字符串，并写入到文件中，各个值之间用逗号分隔
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    #写入换行符，确保每个物体实例的信息在文件中单独占据一行
    list_file.write('\n')

#使用for循环，遍历数据集中每个数据集和每个图像，并调用convert_annotation()函数进行转换处理，将标注信息写入到相应的文件中
for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        convert_annotation(year, image_id, list_file)
    list_file.close()
