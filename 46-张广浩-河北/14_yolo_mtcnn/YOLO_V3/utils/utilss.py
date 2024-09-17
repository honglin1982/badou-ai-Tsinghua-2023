from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    将预测的box还原到原始尺寸，只对钱4位也就是box的值操作
    """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list). 预测正样本 ，iou大于阈值的预测样本 所有类别的正样本标记，还要区分某类别tp,fp  [1,1,0,1,0,1,1] 其中1为正样本
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    1,将正样本、置信度、预测类别按照置信度逆序排序
    2，找到标签所有不同类别，针对每一类别计算指标
        1，获取TP+NP：标签中属于该类别的标签数n_gt
        2，获取TPc，FPc：正样本中属于该类别的数目tpc ， 正样本中不属于该类别的数目fpc
        3，recall = tpc / TP+NP
        4，precision = tpc / TP+FP
        5，计算ap，使用recall，precision计算面积
    3，整理并且返回p（每类的精确度）, r（每类的召回率）, ap, f1, （类别个数）

    """

    # Sort by objectness 按照置信度逆序排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    #对于每一类别
    aps = []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c  #预测为c类的预测结果
        n_gt = (target_cls == c).sum()  # Number of ground truth objects 标签中该类别的数目TP+NP
        n_p = i.sum()  # Number of predicted objects 预测为c类的数目

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            #进行累计求和，表示一个预测时fp，tp 两次预测时fp，tp， 三次预测时 fp tp
            fpc = (1 - tp[i]).cumsum()  #fp tp中为0的数目
            #这个函数的功能是返回给定axis上的累计和
            tpc = (tp[i]).cumsum()   #tp中为1的数目

            # Recall：预测正确该类别数/样本中总体该类别数
            # 召回率，tp/ groudtruth已知的检测框的数量，1 pre一张 /#gt两张#对 -1/2， 2 对pre一张 /#gt两张#  1/2，3 pre两张/#gt两张# 1
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            # 准确率 ，正确1/预测1  1， 正确1/预测2  1/1 ，正确2/预测3  2/3
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))


    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)

    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    # 得到召回率不等的节点，也就是横坐标
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    # 后一项横坐标 m[i+1] - m[i] = 横坐标长度 * 纵坐标 = 面积   所有面积和为ap
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """
    Compute true positives, predicted scores and predicted labels per sample
    :param outputs: (bc,x1, y1, x2, y2, object_conf, class_score, class_pred（id）)
    :param targets: (batch_size, 6)，其中6指的是num, cls, center_x, center_y, widht, height
    :param iou_threshold:
    :return:[true_positives, pred_scores, pred_labels]
    返回[预测正样本标记(pre_num),预测置信度(pre_num),预测标签(pre_num)]
    1，对于每张图片进行处理，获取预测结果位置信息，置信度信息，类别信息
    2，获取所有标签中该图像类别信息annotations，类别，定位
    3，将每张图每一个预测box与所有标签的box对比
        1，如果已统计box数与标准数相同，break
        2，如果预测类别不在标签类别中，直接忽略，视为负样本
        3，如果当前预测box与所有标注box的中iou最大的大于阈值，视为正样本检测框数目加一
    4, 进行整理，[true_positives, pred_scores, pred_labels]
    5，返回正样本标记集合，置信度集合，预测类别集合，都是预测box个数
    sample_metrics - list[8] 每一项 list[0] = [预测正样本标记(pre_num),预测置信度(pre_num),预测标签(pre_num)]

    """

    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue
        #针对一张图预测结果
        output = outputs[sample_i]
        #x1, y1, x2, y2
        pred_boxes = output[:, :4]
        #object_conf
        pred_scores = output[:, 4]
        #类别id
        pred_labels = output[:, -1]
        #TP+FN = 预测结果数（432,7）中432
        true_positives = np.zeros(pred_boxes.shape[0])
        # 针对一张图标记结果
        #把对应ID下的target和图像进行匹配，同一张图的预测结果（40,5）  取cls, center_x, center_y, widht, height
        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        #target_labels类别标签
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            #target_boxes边界框标签（40,4）
            target_boxes = annotations[:, 1:]
            #预测box 和 预测标签合并 (x1, y1, x2, y2,类别id)
            #取每一次的预测结果，box(4),lable(1)
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                #pred_box坐标 ，pred_label类别
                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels如果预测类别不在标签类别中，直接忽略，视为负样本
                if pred_label not in target_labels:
                    continue
                #iou为当前预测结果与所有标注box计算的iou中最大的,box_index(11)第11个标准box被匹配
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                #IOU大于阈值且没被记录过，为预测正样本TP+FP，注意这里仅使用IOU判断样本正负，不考虑类别
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1 #预测结果中该预测结果为正样本（432）正1负0
                    detected_boxes += [box_index] #被检测过的标准框
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    input （b,3*(13x13+26x26+52x52),85） 85 = x,y,w,h,score+80 :c1 c2 c3
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    1,将预测坐标的(x,y,w,h)转化为(x1,y1,x2,y2)
    2,初始化输出列表，列表长度为bc
    3,对每一张图像进行处理
        1，预测结果中置信度con要大于阈值，置信度抑制
        2，按照con*max(class)得到的score排序，得到每个预测结果的预测类别分数和列表编号，整理：(x1, y1, x2, y2, object_conf, class_score, class_pred) (177,7)
        3，非极大值抑制
            1，两个mask，一个取当前score最大box与所有box的IOU>阈值-->预测的为同一位置，一个取类别相同box-->预测同一类别
            2，两个mask合并表示预测同一目标。
            3，取预测该目标的所有box的置信度，按照置信度进行加权平均，综合得到该目标的box位置。
            4，保存该box预测结果，排除预测该目标的所有box，剩下预测其他目标的box继续上述抑制过程。
            5，如果该图存在目标预测结果，则都保存到输出list的。一个图对应一个bc
    4，返回输出结果。
    Returns detections with shape:
        (bc,x1, y1, x2, y2, object_conf, class_score, class_pred类别id)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    #现在的一个prediction里包含batch张图像，一张一张处理
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # 过滤置信度低于阈值的类即背景类(10647,85)->(177->85)
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence使用obc_con * class(i)_con = 每个类别的概率
        #在三个尺度所有网格中选择类别分数最大的，取类别得分最高的类的最大得分(177)
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0] #（1,13*13+26*26+52*52,1）
        # Sort by it
        #按照置信度分数降序排序
        image_pred = image_pred[(-score).argsort()]
        #class_score类别得分(177), class_pred(177)列表id 选择得分最大的作为类别id
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        #(x1, y1, x2, y2, object_conf, class_score, class_pred) (177,7)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        #对每一类别进行nms
        while detections.size(0):
            #拿置信度最高的box1与其他所有计算iou
            #若box1与box2的IOU大于阈值，说明这俩预测同一类的同一物体，则box2从输入列表删除，若低于阈值，说明俩书预测同一类别不同目标
            #比较第一个box与所有box的iou,删除iou>threshold的box,即剔除所有相似box(177)，值为True，False
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            #比较第一个box的类别id（class_pred）和所有box的类别id是否相同，boolean
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            #满足俩条件 1预测类别相同 2iou>阈值=====这些个box预测的是同一目标
            invalid = large_overlap & label_match
            #weights = object_conf（10,1）
            #(x1, y1, x2, y2, object_conf, class_score, class_pred)
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence加权求和，以10个预测同一目标的box，置信度高权重大，确定该目标最终边界。
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            #排除10个预测该目标的box
            detections = detections[~invalid]
        if keep_boxes:
            #把8张图的nms结果合并起来
            #函数stack()对序列数据内部的张量进行扩维拼接，指定维度由程序员选择、大小是生成后数据的维度区间。
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    """
    ignore_thres:4张图片，15x15个网格，每个网格3个不同尺寸的anchor[4,3,15,15]
    b, best_n, gj, gi这四个东西确定标签具体图像具体使用的锚框具体所在的网格。之后的东西都标注这些目标网格的信息
    :param pred_boxes: 预测框 pred_boxes:[4，3，15，15，4] [batch_num,anchors_num,grid_size,grid_size,4(x,y,w,h)]，这个位置是真实在原图中的位置
    :param pred_cls: 类别[4，3，15，15，80] [batch_num,anchors_num,grid_size,grid_size,num_class] 预测的类别
    :param target:  标签值[num_boxes,6] targets(batch_num , calss , x, y ,w ,h)
    :param anchors: 锚框 (3,2)
    :param
    #返回  iou_scores 位置预测与真实偏差iou, class_mask 预测类别正确否, obj_mask 含目标否, noobj_mask 不含目标否, tx, ty, tw, th, #真实位置坐标，tcls #真实类别, tconf #真实置信度
    :return:
    """
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    # 预测参数格式：[4，3，13，13，80] 与 实际标的框对比做计算损失值，要进行格式转化
    nB = pred_boxes.size(0) #batchsize 4
    nA = pred_boxes.size(1) #每个格子对应锚框数量 3
    nC = pred_cls.size(-1) #类别数量 80
    nG = pred_boxes.size(2) #网格大小 15

    # Output tensors
    #mask操作 针对每张图片，每个网格，三个不同尺寸anchor
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0) # obj，某图某网格上某anchor包含物体, 即为1，默认为0 考虑前景 【4,3,13,13】每个网格
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1) # noobj，某图某网格上某anchor包含物体, 即为0，默认为1 考虑背景 【4,3,13,13】
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0) # class   某图某网格上某anchor预测类别正确为1 【4,3,13,13】
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0) #iou分数  某图某网格上某anchor【4,3,13,13】
    tx = FloatTensor(nB, nA, nG, nG).fill_(0) #包含目标网格x偏移
    ty = FloatTensor(nB, nA, nG, nG).fill_(0) #包含目标网格y偏移
    tw = FloatTensor(nB, nA, nG, nG).fill_(0) #包含目标网格w偏移
    th = FloatTensor(nB, nA, nG, nG).fill_(0) #包含目标网格h偏移
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0) #包含目标网格类别【4,3,13,13,80】

    # Convert to position relative to box
    # 【x y w h】target中的xywh都是0-1的，是比例，可以得到其在当前gridsize上的xywh 真实值
    # gxy具体到哪一个网格 7,4
    target_boxes = target[:, 2:6] * nG #一个bc所有目标的位置 （31,4）
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    # 3种每一种规格的anchor跟每个标签上的框的IOU得分
    # best_n确定每个网格选择的锚框
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors]) #（3,31）
    best_ious, best_n = ious.max(0) # best_iou(31)3个网格中iou最好的那个  best_n(31)anchors序号
    # Separate target values
    # b确定每个网格对应的bc
    # 拿到第几张图片和类别
    # gi, gj确定每个目标所在具体的网格
    b, target_labels = target[:, :2].long().t() # 真实框所对应的batch，以及每个框所代表的实际类别
    gx, gy = gxy.t()#gx(31)标签中心x gy(31)标签中心y
    gw, gh = gwh.t()
    gi, gj = gxy.long().t() #位置信息，向下取整了
    # Set masks----------------------------------------------------------------------------
    #我们知道了对于每个gt框，应该在哪个cell，由哪个anchors来负责取整。对于最大的anchor对应框设为有样本
    obj_mask[b, best_n, gj, gi] = 1 # 1表示有obj，0表示没有
    noobj_mask[b, best_n, gj, gi] = 0 # 1表示没有obj，0表示有

    # Set noobj mask to zero where iou exceeds ignore threshold
    # 在不包含目标mask中若iou大于阈值，置零，说明这些2个anchor与目标框IOU较大，也认为存在目标
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0
    #--------------------------------------------------------------------------------------
    #tx ty大部分为0，其中gj,gi在的也就是目标存在的网格值不为0，为偏移，比如tx[0,0,15,21]=0.4 ty[0,0,15,21]=0.5
    # Coordinates 坐标转换---》x,y 在网格中真实位置-网格左上角坐标=x,y偏移 0张图，0号锚框，5,21网格x,y偏移值
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height 坐标转换---》真实值转化为相对于网格的坐标
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # --------------------------------------------------------------------------------------
    # One-hot encoding of label #将真实框的标签转换为one-hot编码形式
    #tcls[0,0,15,21,67]=1 0张图，0号锚框，15,21网格目标是类别id为67
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    # 计算真实值与预测值之间的情况（算对/错） 类别预测结果
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    # 与真实框想匹配的预测框之间的iou值  位置预测结果[0,0,15,21]预测结果与所有目标box做iou
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float() # 真实框的置信度，也就是1
    #返回  iou_scores 位置预测与真实偏差iou, class_mask 预测类别正确否, obj_mask 含目标否, noobj_mask 不含目标否, tx, ty, tw, th, #真实位置坐标，tcls #真实类别, tconf #真实置信度
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
