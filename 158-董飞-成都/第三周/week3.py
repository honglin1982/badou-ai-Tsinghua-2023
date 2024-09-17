# -*- coding=utf-8 -*-
# 1.实现最临近插值 2.实现双线性插值 3.证明几何中心对称系数 4.实现直方图均衡化
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


def proximity_interpolation(img: np.array = None, new_shape: tuple = None) -> np.array:
    """
    最临近插值函数

    parameter :
        img:灰度图或三通道图的numpy矩阵
        new_shape:图像经过resize后的大小
    return :
        输出resize之后的numpy矩阵
    """
    # 获取原图的shape
    original_shape = img.shape
    # 如果原图是多通道的，那么新图也是多通道的
    new_shape = (new_shape[0], new_shape[1], original_shape[2]) if len(
        original_shape) == 3 else new_shape
    # 创建初始画板，大小与new_shape一致，值全初始化为0,即黑色
    img_upsampling = np.zeros(new_shape, dtype=np.uint8)  # 创建初始out_img
    # 对新图的每一个像素点进行遍历，根据新图像素点的位置去原图找相应的临近像素点值
    for y in range(new_shape[0]):
        for x in range(new_shape[1]):
            # 根据坐标在新图上的比例，按照比例去找原图上该比例处的像素点的值
            img_upsampling[y, x] = img[int(y/new_shape[0]*original_shape[0]+0.5),
                                       int(x/new_shape[1]*original_shape[1]+0.5)]
   
    return img_upsampling

def bilinear_interpolation(img: np.array = None, new_shape: tuple = None)->np.array:
    """
    双线性插值函数

    parameter :
        img:灰度图或三通道图的numpy矩阵
        new_shape:图像经过resize后的大小
    return :
        输出resize之后的numpy矩阵
    """
    # 获取原图的shape
    original_shape = img.shape
    # 如果原图是多通道的，那么新图也是多通道的
    new_shape = (new_shape[0], new_shape[1], original_shape[2]) if len(
        original_shape) == 3 else new_shape
    # 创建初始画板，大小与new_shape一致，值全初始化为0,即黑色
    img_upsampling = np.zeros(new_shape, dtype=np.uint8)  # 创建初始out_img
    # 对新图的每一个像素点进行遍历，根据新图像素点的位置去原图找相应的临近像素点值
    for y in range(new_shape[0]):
        for x in range(new_shape[1]):
            #找到目标图的坐标在原图的位置,根据比例去寻找,并按照公式推算进行几何中心重合
            src_y=(y+0.5)/new_shape[0]*original_shape[0]-0.5
            src_x=(x+0.5)/new_shape[1]*original_shape[1]-0.5
            #根据原图的坐标找到原图坐标临近四点的x,y坐标
            x0,y0=int(src_x),int(src_y) #int即向下取整，比目标坐标点坐标值更小的坐标点
            x1,y1=min(x0+1,original_shape[1]-1),min(y0+1,original_shape[0]-1)  #超出边界只能取边界大小
            #根据临近四点的x,y坐标，取临近四点的像素值
            x0_y0,x1_y0,x0_y1,x1_y1=img[y0,x0],img[y0,x1],img[y1,x0],img[y1,x1]
            #根据临近四点的像素值和插值公式计算辅助插值虚拟点p1、p2的像素值
            p1=(x1-src_x)*x0_y0+(src_x-x0)*x1_y0
            p2=(x1-src_x)*x0_y1+(src_x-x0)*x1_y1
            #根据辅助插值虚拟点的像素值和公司计算目标像素点的像素值
            img_upsampling[y,x]=(y1-src_y)*p1+(src_y-y0)*p2

    return img_upsampling.astype(np.uint8)

def my_equalizeHist(img: np.array = None) -> np.array:
    """
    直方图均衡化函数

    parameter :
        img:灰度图或三通道图的numpy矩阵
    return :
        输出直方图均衡化之后的numpy矩阵
    """
    # 如果是多通道图对每个通道单独进行处理再合并通道
    if len(img.shape) == 3:  # 传入的如果是多通道的彩色图
        return cv2.merge(  # 通道合并
            [my_equalizeHist(img[:, :, i]) for i in range(img.shape[2])])  # 对每一个通道单独做直方图均衡
    # 统计每个像素级的像素点个数
    pixel_value, pixels_number = np.unique(
        img, return_counts=True)
    # 累加像素点个数，该像素级及该像素级之前的所有像素级像素点个数总数
    pixels_number_accumulation = [
        pixels_number[:i+1].sum() for i in range(len(pixels_number))]
    # 累加像素点个数与总数的比值，计算出总比值后即可计算映射像素值
    pixels_accumulation_ratio = pixels_number_accumulation / \
        pixels_number_accumulation[-1]
    # 像素映射字典，键值对{某个像素级:该像素级在直方均衡化后对应的新像素级},
    # 新映射的像素值该等同于该比值*256-1 （0~255，一共256个像素级，
    # 减一是因为像素值是从0开始的，这里乘法乘出来的是第几个像素级，比如第256个像素级，它的像素值是像素级减一也就是255)
    mapping_dict = dict(
        zip(pixel_value, (pixels_accumulation_ratio*256-1).astype(np.uint8)))
    # 根据像素映射字典对输入图像的像素值进行替换,先copy一份img，因为在for的执行过程中矩阵的值是不断变化的，而索引必须对原图的值进行索引
    return_img = img.copy()
    for value in pixel_value:
        return_img[img == value] = mapping_dict[value]
    return return_img

if "__main__" == __name__:
    img = cv2.imdecode(np.fromfile('1.jpg', dtype=np.uint8), 1)  # 读入灰度图
    fig = plt.figure(figsize=(6, 6))  # 创建画板

    #BGR图转灰度图后，调用opencv的直方图均衡化函数，用于对比自己写的直方图均衡化函数
    img_cvequalizeHist = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('opencv_equalizeHist')
    ax1.imshow(img_cvequalizeHist,cmap='gray')

    # BGR图转灰度图后，调用自己写的直方图均衡化函数（自己写的支持传入多通道图像）
    img_myequalizeHist = my_equalizeHist(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    ax2 = fig.add_subplot(2, 2, 2)  # 创建子画板
    ax2.set_title('my_equalizeHist')  # 设置子画板标题
    ax2.imshow(img_myequalizeHist,cmap='gray')  # 绘图

    # 调用自己写的临近插值函数
    my_proximity_interpolation = proximity_interpolation(img, (300, 400))
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('my_proximity_interpolation')
    ax3.imshow(cv2.cvtColor(my_proximity_interpolation,cv2.COLOR_BGR2RGB))

    #调用自己写的双线性插值函数
    my_bilinear_interpolation=bilinear_interpolation(img,(300,400))
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.set_title('my_bilinear_interpolation')
    ax3.imshow(cv2.cvtColor(my_bilinear_interpolation,cv2.COLOR_BGR2RGB))

    plt.show()  # 显示