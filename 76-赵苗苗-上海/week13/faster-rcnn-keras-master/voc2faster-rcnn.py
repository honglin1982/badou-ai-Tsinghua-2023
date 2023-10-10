
#代码的整体功能是将一个数据集的标记文件进行划分，生成训练集、验证集和测试集，并将它们的文件名保存到特定的文本文件中
"""
1.读取指定路径下的所有文件(XML文件)
2.计算训练集，验证集和测试集所占总数据集的比例
3.随机选择索引来划分训练集，验证集和测试集
4.创建用于存储划分结果的四个文本文件
5.遍历数据集的索引列表，将对应的文件名写入相应的文本文件中
6.关闭打开的文本文件，确保写入操作被保存
"""
import os
import random   #生成随机数

#定义了标记文件路径和保存划分结果的路径
xmlfilepath=r'D:\AI\submit assignment\Assignment\week13\faster-rcnn-keras-master\VOCdevkit\VOC2007\Annotations'
saveBasePath=r"D:\AI\submit assignment\Assignment\week13\faster-rcnn-keras-master\VOCdevkit\VOC2007/ImageSets/Main/"

#设定了训练集和验证集在总数据集所占的比例，默认情况下，训练集和验证集各占总数据集的100%
trainval_percent=1
train_percent=1

#从指定的标记文件路径中读取所有的XML文件，并将它们存储在total_xml列表中
temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

#计算出要划分的训练集，验证集和测试集的大小
num=len(total_xml)  
list=range(num)  
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  

#使用random.sample()函数从数据集索引的列表中随机选择训练集，验证集和测试集的索引
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  

#打印出训练集和验证集的大小
print("train and val size",tv)
print("traub suze",tr)

#创建了四个文本文件，用于保存划分后的训练集，验证集和测试集的文件名
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  

#for循环遍历数据集索引的列表，将每个索引对应的XML文件名写入相应的文本文件中
#训练集和验证集的文件名都写入了trainval.txt文件中，测试集的文件名单独写入到了test.txt文件中
#训练集的文件名还会写入train.txt文件中，验证集的文件名则写入到val.txt文件中
for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  

#关闭了创建的文本文件，确保写入操作被保存
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()
