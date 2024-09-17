# -*- coding: utf-8 -*-
# @Author  : ZGH
# @Time    : 2022/11/29 10:53
# @File    : test.py
# @Software: PyCharm


from __future__ import division

from models import *
from utils.utilss import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
#1.定义evaluate评估函数
#2.解析输入的参数
#3.打印当前使用的参数
#4.解析评估数据集的路径和class_names
#5.创建model
#6.加载模型的权重
#7.调用evaluate评估函数得到评估结果
#8.打印每一种class的评估结果ap
#9.打印所有class的平均评估结果mAP

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    """

    :param model: 模型
    :param path: 测试图像数据路径 data/coco/5k.txt
    :param iou_thres: IOU阈值
    :param conf_thres: 置信度阈值
    :param nms_thres: 非极大值抑制阈值
    :param img_size: 图像尺寸
    :param batch_size: 组个数
    :return:precision精度, recall召唤率, AP, f1, ap_class类别

    1,构建数据集
        1，初始化图像路径列表，标签路径列表
        2，获取图像并进行pading，获取标签进行整理targets: num_obj , classes , x , y , w ,h
    2，对于每一批图像进行模型预测、非极大值抑制、正样本统计
        1，模型预测，outputs = model(imgs)，图像bc输入model，进行边界框预测，返回（b,3*(13x13+26x26+52x52),85）
        2，非极大值抑制，通过con阈值排除con低的预测结果，
           通过IOU阈值，保留score最高且预测某目标的box，抑制预测同一目标且score低的box。
           返回一组图像中每个图像抑制后预测结果output[{0:[432,7]},{1:[85,7]},……{7:[20,7]}]，且按照score=con*class 排序
        3，正样本统计，记录前标准数目的（target_num）预测结果与标准结果IOU大于阈值的预测结果为正样本。
           返回sample_metrics - list[8] 每一项 list[0] = [预测正样本标记(pre_num),预测置信度(pre_num),预测标签(pre_num)]
    3，计算评估指标：precision, recall, AP, f1, ap_class
    4，返回precision, recall, AP, f1, ap_class
    """
    model.eval()

    # Get dataloader加载数据
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    #tqdm 进度条
    #评估第i批batch,在 这里取一个batch的时候，对于batch进行处理，target中每张图的0号位置与其batch_i对应。collate_fn
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels 获取标签
        # targets:  (batch_size, 6)，其中6指的是num, cls, center_x, center_y, widht, height，其中
        # num指的第num个图片
        labels += targets[:, 1].tolist()
        # Rescale target 标签类型转化 中心长宽->左上右下
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        #调整为原图大小
        targets[:, 2:] *= img_size
        #输入图像转化为tensor
        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            #向模型输入图像得到输出,此时target没传=None，不进行loss计算
            outputs = model(imgs)
            #（b,3*(13x13+26x26+52x52),85）
            #非极大值抑制output[{0:[432,7]},{1:[85,7]},……{7:[20,7]}]
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            #(bc , x1, y1, x2, y2, object_conf, class_score, class_pred（id）)
            #output.shape(bc,7)  (x1, y1, x2, y2, object_conf, class_score, class_pred)

        #返回true_positives正样本id, pred_scores得分, pred_labels类别id    得到测试样本的各项指标
        #sample_metrics - list[8] 每一项 list[0] = [预测正样本标记(pre_num),预测置信度(pre_num),预测标签(pre_num)]
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    #计算tp
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    #计算pre rec ap
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    #MAP=np.mean(AP)
    #print('---map {}---'.format(np.MAP))
    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default=r"D:\Program Files (x86)\models\yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    #得到测试图像路径文件 data/coco/5k.txt
    #这里对于各个图像路径进行一个修改 ，在5k.txt文件中，图像路径改为图像在本地存储路径
    valid_path = data_config["valid"]
    #得到类名称
    class_names = load_classes(data_config["names"])

    # Initiate model初始化模型，指定模型结构文件
    model = Darknet(opt.model_def).to(device)
    #如果预训练参数文件后缀为.weights，加载预训练模型
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")


