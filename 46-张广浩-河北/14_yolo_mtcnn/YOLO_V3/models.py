# -*- coding: utf-8 -*-
# @Author  : ZGH
# @Time    : 2022/11/14 17:33
# @File    : models.py
# @Software: PyCharm
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utilss import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchsummary import summary


def create_modules(module_defs):
    '''

    :param module_defs: 模型参数
    :return:
    '''
    hyperparams = module_defs.pop(0) #获取网络信息，conv block之前信息 {'type': 'net', 'batch': '16', 'subdivisions': '1', 'width': '416', 'height': '416', 'channels': '3', 'momentum': '0.9', 'decay': '0.0005', 'angle': '0', 'saturation': '1.5', 'exposure': '1.5', 'hue': '.1', 'learning_rate': '0.001', 'burn_in': '1000', 'max_batches': '500200', 'policy': 'steps', 'steps': '400000,450000', 'scales': '.1,.1'}
    # 输入通道数
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    # module_i层数 module_def参数
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        # 构建卷积层 Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            # 构建批归一化
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            # 构建激活函数
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            # 构建池化层
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)
            # 上采样
        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)
            #深度方向堆叠，也就是cont操作，通道数堆叠 在forward进行，这里先创建空层
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())
            # 残差操作 残差操作通道数不变，每个通道特征图相加，也就是add操作在forward进行，这里先创建空层
        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())
            # 构建yolo检测头 总共三个，分别在13x13  26x26  52x52 三个不同尺度构建
        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")] #6,7,8
            # Extract anchors 获取对应层次的锚框尺寸，如对于678 拿anchors中6 7 8 后三个大尺寸锚框
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
                # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        """
        F.interpolate
        利用插值方法，对输入的张量数组进行上\下采样操作，换句话说就是科学合理地改变数组的尺寸大小，尽量保持数据完整。
        input(Tensor)：需要进行采样处理的数组。
        size(int或序列)：输出空间的大小
        scale_factor(float或序列)：空间大小的乘数
        mode(str)：用于采样的算法。'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'。默认：'nearest'
        :param scale_factor:
        :param mode:
        """
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """检测层
    yolo_layer = YOLOLayer(anchors, num_classes, img_size)
    返回：(bc,grid_n * grid_n,85)
    (85 = bx,by,bw,bh,con,c1,c2,c3……,c80)
    """

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.img_dim = img_dim
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()  # 均方损失 L2损失
        self.bce_loss = nn.BCELoss()  # 单标签二值损失 0-1之间
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid 计算每个网格的偏移量（在预测的相对位置数值基础上应该＋的网格数）
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        # Tensors for cuda support
        #print(x.shape)  # 打印当前参数结果的信息：示例[4，255，15，15] ：[batch数值，特征图个数，当前特征图的大小15*15] 4组所有当前特征图所有网格预测结果
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor  # 带.cuda说明使用GPU训练、使用CPU把.cuda去掉即可
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim  # 输入图像的大小
        num_samples = x.size(0)  # 当前的batch数值，一次训练几张图像
        grid_size = x.size(2)  # 网格大小=（输入图像大小/（2**5=32））

        prediction = (  # 预测的结果
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                # （num_samples样本个数，num_anchors 候选框个数、num_classes + 5 [80类+ 4（x,y,w,h） +1(confidence)]，锚框个数，锚框个数）
                .permute(0, 1, 3, 4, 2)  # 参数维度变换
                #（num_samples样本个数，num_anchors 候选框个数，锚框个数，锚框个数，num_classes + 5 [80类+ 4（x,y,w,h） +1(confidence)]）
                # （num_samples，num_anchors，grid_size，grid_size，num_classes + 5 [80类+ 4（x,y,w,h） +1(confidence)]）
                # (bc,3,85,13,13)->（bc,3,13,13,85）
                .contiguous()
        )
        #print(prediction.shape)  # 例：维度变换之后的结果：[4，3，15，15，85]，其中85参数顺序：[x,y,w,h,c,类别。。。。。。]
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # 预测值中间坐标偏移 Center x，
        y = torch.sigmoid(prediction[..., 1])  # 预测值中间坐标偏移 Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        # 且此时得到的x,y,w,h都是介于（0，1）之间的数值，是预测中心点与网格点之间相对位置，要得到特征图中的实际位置，还要对位置坐标进行还原
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf 置信度[4，3，15，15，1]
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred. 预测的类别可能性

        # If grid size does not match current we compute new offsets  如果网格大小与当前不匹配，则计算新的偏移量
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)  # 相对位置得到对应的绝对位置比如之前的位置是0.6,0.6变为 8.6，8.6这样的

        # Add offset and scale with anchors #  得到特征图中的实际位置，使用相对偏移与网格左上角点相加
        # pred_boxes:[4，3，15，15，4]
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        #这里self.为tx ty tw th 看图YOLOv3图片
        #x.data为cx y.data为cy  self.grid_x=1  self.anchor_w为pw锚框宽  self.anchor_h为ph锚框高
        #用于计算bx by bw bh box在网格的位置
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        #在最后一个维度进行堆叠，(bc,n*n,4+1+80)
        output = torch.cat(
            (  # 预测值是在特征图的基础上的值，输出值应该在原始图像上，还要进行还原
                pred_boxes.view(num_samples, -1, 4) * self.stride,  # 还原到原始图中（ × 32 ） 位置坐标
                pred_conf.view(num_samples, -1, 1),  # 置信度分数
                pred_cls.view(num_samples, -1, self.num_classes),  # 类别分数
            ),
            -1,
        )
        # 计算损失值
        if targets is None:
            return output, 0
        else:
            # iou最大为正样本，小于阈值为负样本，其他忽略
            # 返回 ：
            # iou_scores 位置预测与真实偏差iou, class_mask 预测类别正确否, obj_mask 含目标否,
            # noobj_mask 不含目标否, tx, ty, tw, th, #真实位置坐标，tcls #真实类别, tconf #含有目标mask
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
            obj_mask = obj_mask.bool()
            noobj_mask = noobj_mask.bool()
            # iou_scores(4,3,30,30)：真实值与最匹配的anchor的IOU得分值 class_mask(4,3,30,30)：分类正确的网格索引  obj_mask：目标框所在位置的最好anchor置为1 noobj_mask obj_mask那里置0，还有计算的iou大于阈值的也置0，其他都为1 tx, ty, tw, th, 对应的对于该大小的特征图的xywh目标值也就是我们需要拟合的值 tconf 目标置信度
            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            #--------------------------------------------------------------------------------------
            # 正样本定位损失，使用均方误差，只计算包含包含目标的网格(4,3,15,15)  31值与31值比较
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])  # 只计算有目标的损失
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            # --------------------------------------------------------------------------------------
            # 正样本置信度损失二值损失(4,3,15,15) 与1比较  31值与31值
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])  # loss_conf_obj  前景的损失
            # 负样本置信度损失(4,3,15,15) 与0比较
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])  # loss_conf_noobj  背景的损失
            # 总体置信度损失 = r1*正样本置信度损失 + r2*负样本置信度损失
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj  # 有物体越接近1越好 没物体的越接近0越好
            # 正样本类别损失，（4,3,15,15,80）二元交叉熵 与1比较
            # --------------------------------------------------------------------------------------
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])  # 分类损失
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls  # 总损失

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()  # 31分类得分
            conf_obj = pred_conf[obj_mask].mean()  #31个 正样本置信度得分
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()  # 正样本且置信度>50为正例  tp+fp
            iou50 = (iou_scores > 0.5).float()  # IOU大于50
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf  # 置信度高*分类正确*含有目标
            # precision = tp / tp + fp         tp=置信度>50 +分类正确+含有目标+ iou>50   预测对/预测对+预测错
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            # recall = tp / tp+fn     预测对/标签对+标签错
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }
        return output, total_loss


class Darknet(nn.Module):
    def __init__(self, config_path, img_size=416):
        """
        :param config_path: 网络参数文件路径
        :param img_size: 图像大小
        """
        super(Darknet, self).__init__()
        #获取模型结构参数
        self.module_defs = parse_model_config(config_path)
        # 1 卷积
        # 1 64Residual Bolock   ---------->416->208缩小2倍图像     ---->208x208x64
        # 2 128Residual Bolock  ---------->416->104缩小4倍图像     ---->104x104x128
        # 8 256Residual Bolock   ---------->416->52缩小8倍图像     ---->52x52x256+++++++++++++++++++++++++++++++++++++++++++++++(26x26x128->52x52x128)---->(52x52x384)---->c2db5l---->(52x52x128)--->yolo31--(52x52x256)-->52x52x255
        # 8 512Residual Bolock   ---------->416->26缩小16倍图像    ---->26x26x512+++++（13x13x256->26x26x256）--->(26x26x768)--c2db5l-->(26x26x256)|up|----->yolo31--26x26x512--->26x26x255
        # 4 1024Residual Bolock   ---------->416->13缩小32倍图像   ---->13x13x1024--c2db5l-->13x13x512 |up| ----13x13x1024--yolo31-->13x13x255
        #
        #hyperparams 网络超参数信息  module_list网络列表信息,按照模型参数文件构建模型基本架构，残差和特征金字塔堆叠先置为空层
        #
        #
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        #hasattr 判断对象是否包含某属性 通过metrics属性检测三个yolo检测层
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]  # 一共3层
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        #forward 参数注意，一个为图像数据，一个为标签数据，别加其他的
    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        #zip 打包为元组列表a = [1,2,3] b = [4,5,6]
        #zipped = zip(a,b)->[(1, 4), (2, 5), (3, 6)]
        #这里将网络构架名称与网络构架对应
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            #卷积，池化，上采样直接进行
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            #route操作，获取之前的层，layers = -1, 61，使用指定层堆叠，
            #使用cat进行深度方向维度拼接，26x26x512+++++（13x13x256->26x26x256）--->(26x26x768)
            elif module_def["type"] == "route":
                #route两个层在深度维度进行cat
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)  # 特征拼接操作
            #shortcut 残差操作 两层进行add
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"]) #-3
                x = layer_outputs[-1] + layer_outputs[layer_i]  # 残差连接操作： 加法的操作
            elif module_def["type"] == "yolo":
                #yolo检测头，返回(bc,grid_n * grid_n, 85)  loss
                x, layer_loss = module[0](x, targets, img_dim)  # 输入的x是前一层的结果；targets 标签信息 ； img_dim 输入图像的大小
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        #在网格维度进行堆叠，(bc,13x13,85)+(bc,26x26,85)+(bc,52x52,85)=(bc,13x13+26x26+52x52,85)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            #module_def["type"]是层名称
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

def get_test_input():
    img = cv2.imread("train.jpg")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

if __name__ == '__main__':
    model_def = "D:\python\YOLO\YOLO_V3\config\yolov3.cfg"
    model = Darknet(model_def)
    inp = get_test_input()
    pred = model(inp)
    print(pred.shape)
    # 3 * (13*13+26*26+52*52)=10647个网格
    # (bc,13x13,85)+(bc,26x26,85)+(bc,52x52,85)=(bc,13x13+26x26+52x52,85)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    #summary(model,(3,416,416))


    print(model)


