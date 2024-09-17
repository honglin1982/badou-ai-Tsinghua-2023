import tensorflow as tf
import numpy as np
from PIL import Image


def loadWeights(varList, weightsFile):
    """
    Introduction
    ------------
        加载预训练好的darknet53权重文件
    Parameters
    ----------
        varList: 赋值变量名
        weightsFile: 权重文件
    Returns
    -------
        assignOps: 赋值更新操作
    """
    with open(weightsFile, 'rb') as f:

        _ = np.fromfile(f, dtype=np.int32, count=5) #先读出前5个前缀
        weights = np.fromfile(f, dtype=np.float32)

    ptr = 0
    i = 0
    assignOps = []
    while i < len(varList) - 1:
        var1 = varList[i]
        var2 = varList[i + 1]
        if 'conv2d' in var1.name.split('/')[-2]:
            if 'batch_normalization' in var2.name.split('/')[-2]:
                gamma, beta, mean, var = varList[i + 1 : i + 5]
                batchNormVars = [beta, gamma, mean, var]
                for var in batchNormVars:
                    shape = var.shape.as_list()
                    numParams = np.prod(shape) #按shape相乘
                    varWeights = weights[ptr:ptr + numParams].reshape(shape)
                    ptr += numParams
                    assignOps.append(tf.assign(var, varWeights, validate_shape=True))
                i += 4
            elif 'conv2d' in var2.name.split('/')[-2]:
                #load biases
                bias = var2
                biasShape = bias.shape.as_list()
                biasParams = np.prod(biasShape)
                biasWeights = weights[ptr : ptr + biasParams].reshape(biasShape)
                ptr += biasParams
                assignOps.append(tf.assign(bias, biasWeights, validate_shape=True))
                #加载完成一个变量
                i += 1
            #加载卷积层权重
            shape = var1.shape.as_list()
            numParams = np.prod(shape)

            varWeights = weights[ptr : ptr + numParams].reshape((shape[3], shape[2], shape[0], shape[1]))
            #转成按列排列
            varWeights = np.transpose(varWeights, (2, 3, 1, 0))
            ptr += numParams
            assignOps.append(tf.assign(var1, varWeights, validate_shape=True))
            i += 1
    return assignOps

def letterboxImage(image, size):
    """
    Introduction
    ------------
        对预测输入图像进行缩放，按照长宽比进行缩放，不足的地方进行填充
    Parameters
    ----------
        image: 输入图像
        size: 图像大小
    Returns
    -------
        boxed_image: 缩放后的图像
    """
    imageW, imageH = image.size
    w, h = size
    newW = int(imageW * min(w * 1.0 / imageW, h * 1.0 / imageH))
    newH = int(imageH * min(w * 1.0 / imageW, h * 1.0 / imageH))
    resizedImage = image.resize((newW, newH), Image.BICUBIC)
    boxedImage = Image.new('RGB', size, (128, 128, 128))
    boxedImage.paste(resizedImage, ((w - newW)//2, (h - newH)//2))
    return boxedImage

def drawBox(image, bbox):
    """
    Introduction
    ------------
        通过tensorboard把训练数据可视化（画框）
    Parameters
    ----------
        image: 训练数据图片
        bbox: 训练数据图片中标记box坐标
    """
    xmin, ymin, xmax, ymax, label = tf.split(value = bbox, num_or_size_splits = 5, axis = 2)
    height = tf.cast(tf.shape(image)[1], tf.float32)
    weight = tf.cast(tf.shape(image)[2], tf.float32)
    newBBox = tf.concat([tf.cast(ymin, tf.float32) / height, tf.cast(xmin, tf.float32) / weight,
                         tf.cast(ymax, tf.float32) / height, tf.cast(ymax, tf.float32) / weight], 2)
    newImage = tf.image.draw_bounding_boxes(image, newBBox)
    tf.summary.image('input', newImage)

def vocAp(rec, prec):
    """
        数据集相关处理
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0)    # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1): mpre[i] = max(mpre[i], mpre[i + 1])
    iList = []
    for i in range(1, len(mrec)) :
        if mrec[i] != mrec[i - 1]: iList.append(i)
    ap = 0.0
    for i in iList : ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre
