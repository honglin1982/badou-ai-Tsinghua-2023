# _*_ coding : utf-8 _*_
# @Time : 2023/8/8 14:56
# @Author : weixing
# @FileName : train
# @Project : cv

import os
from util.datasetUtils import data_ide, voc_annotation
import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from util.FRCNNDataset import FRCNNDataset
from util.FasterRCNNTrainer import FasterRCNNTrainer
from nets.Faster_R_CNN import FasterRCNN
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable

sets = [('last_day', 'train'), ('last_day', 'val'), ('last_day', 'test')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
           "tvmonitor"]
path = os.path.dirname(os.path.abspath(__file__))
xmlfilepath = path + '/VOCdevkit2007/VOC2007/Annotations/'
imagePath = path + '/VOCdevkit2007/VOC2007/JPEGImages/'
saveBasePath = path + '/dataset/'
modelSavePath = path + '/model/'


def build_dataset():
    trainval_perscent = 0.9
    train_persent = 0.8
    num_data = None
    data_ide(xmlfilepath, saveBasePath, trainval_perscent, train_persent, num_data)
    voc_annotation(sets, classes, imagePath, saveBasePath, xmlfilepath)


def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = np.array(images)
    return images, bboxes, labels


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_ont_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    val_total_loss = 0

    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:

        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            imgs, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor)).cuda()
                else:
                    imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
            losses = train_util.train_step(imgs, boxes, labels, 1)
            rpn_loc, rpn_cls, roi_loc, roi_cls, total = losses
            total_loss += total.item()
            rpn_loc_loss += rpn_loc.item()
            rpn_cls_loss += rpn_cls.item()
            roi_loc_loss += roi_loc.item()
            roi_cls_loss += roi_cls.item()

            pbar.set_postfix(**{'total': total_loss / (iteration + 1),
                                'rpn_loc': rpn_loc_loss / (iteration + 1),
                                'rpn_cls': rpn_cls_loss / (iteration + 1),
                                'roi_loc': roi_loc_loss / (iteration + 1),
                                'roi_cls': roi_cls_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size:
                break
            imgs, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor)).cuda()
                else:
                    imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))

                train_util.optimizer.zero_grad()
                losses = train_util.forward(imgs, boxes, labels, 1)
                _, _, _, _, val_total = losses

                val_total_loss += val_total.item()
            pbar.set_postfix(**{'total_loss': val_total_loss / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: {} || Val Loss: {}'.format(total_loss / (epoch_size + 1),
                                                  val_total_loss / (epoch_size_val + 1)))
    print('Saving state, iter:', str(epoch + 1))
    torch.save(net.state_dict(),
               'vocdataset/Epoch{}-Total_Loss{}-Val_Loss{}.pth'.format(epoch + 1, total_loss / (epoch_size + 1),
                                                                       val_total_loss / (epoch_size + 1)))







Cuda = False
NUM_CLASSES = len(classes)
input_shape = [800, 800, 3]
backbone = "vgg"
annotation_path_train = os.path.join(saveBasePath, 'train.txt')
annotation_path_val = os.path.join(saveBasePath, 'val.txt')
annotation_path_test = os.path.join(saveBasePath, 'test.txt')

# 加载预训练主干网络
model_path = os.path.join(modelSavePath, 'model_13.pth')

model = FasterRCNN(NUM_CLASSES, backbone=backbone)
print('Loading weights into state dict...')
device = torch.device('cuda' if (torch.cuda.is_available() and Cuda) else 'cpu')
model_dict = model.state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print('Finished!')

net = model.train()
if Cuda:
    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    net = net.cuda()

with open(annotation_path_train) as f, open(annotation_path_val) as f_v, open(annotation_path_test) as f_t:
    lines_train = f.readlines()
    lines_val = f_v.readlines()
    lines_test = f_t.readlines()
np.random.seed(1)
np.random.shuffle(lines_train)
np.random.shuffle(lines_val)
np.random.shuffle(lines_test)
np.random.seed(None)

num_train = len(lines_train)
num_val = len(lines_val)
train_file = lines_train[:]
val_file = lines_val[:]
test_file = lines_test[:]

# 主干特征提取网络特征通用，冻结训练可以加快训练速度, 也可以在训练初期防止权值被破坏.
# Init_Epoch为起始世代
# Freeze_Epoch为冻结训练的世代
# Epoch总训练世代
if True:
    lr = 1e-4
    Batch_size = 2
    Init_Epoch = 0
    Freeze_Epoch = 50

    optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    train_dataset = FRCNNDataset(train_file, (input_shape[0], input_shape[1]), is_train=True)
    val_dataset = FRCNNDataset(val_file, (input_shape[0], input_shape[1]), is_train=False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=False,
                     drop_last=True, collate_fn=frcnn_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=False,
                         drop_last=True, collate_fn=frcnn_dataset_collate)

    epoch_size = num_train // Batch_size
    epoch_size_val = num_val // Batch_size

    # 冻结一部份训练
    for param in model.extractor.parameters():
        param.requires_grad = False

    # 冻结BN 层
    model.freeze_bn()

    train_util = FasterRCNNTrainer(model, optimizer)
    # print('Starting to train with freeze BN...')
    for epoch in range(Init_Epoch, Freeze_Epoch):
        fit_ont_epoch(net, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
        lr_scheduler.step()

if True:
    lr = 1e-5
    Batch_size = Batch_size
    Freeze_Epoch = 50
    Unfreeze_Epoch = 100

    optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    train_dataset = FRCNNDataset(train_file, (input_shape[0], input_shape[1]), is_train=True)
    val_dataset = FRCNNDataset(val_file, (input_shape[0], input_shape[1]), is_train=False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=False,
                     drop_last=True, collate_fn=frcnn_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=False,
                         drop_last=True, collate_fn=frcnn_dataset_collate)

    epoch_size = num_train // Batch_size
    epoch_size_val = num_val // Batch_size

    # 解冻后训练
    for param in model.extractor.parameters():
        param.requires_grad = True

    # 冻结BN 层
    model.freeze_bn()

    for epoch in range(Freeze_Epoch, Unfreeze_Epoch, ):
        fit_ont_epoch(net, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
        lr_scheduler.step()


