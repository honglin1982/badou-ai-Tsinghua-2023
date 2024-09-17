import os
import random


xmlfilepath = r'./VOCdevkit/VOC2007/Annotations'
saveBasePath = r"./VOCdevkit/VOC2007/ImageSets/Main/"
trainval_pencent = 1
train_pencent = 1

temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith('.xml'):
        total_xml.append(xml)

num = len(total_xml)
list = range(num)
tv = int(num*trainval_pencent)
tr = int(tv*train_pencent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
print('train and val size', tv)
print('traub suze', tr)
ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

for i in list:
    name = total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
