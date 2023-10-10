#这段代码用来生成用于目标检测算法的锚框。锚框是一些预定义的框，用于在图像上进行目标检测
import numpy as np
import keras
import tensorflow as tf
from utils.config import Config
import matplotlib.pyplot as plt
#初始化配置
config=Config()
#生成锚框函数：根据给定的尺寸和宽高比生成一组锚框
def generate_anchors(sizes=None,ratios=None):
    #初始化参数
    if sizes is None:
        sizes=config.anchor_box_scales
    if ratios is None:
        ratios=config.anchor_box_ratios
    #计算锚框的总数
    num_anchors=len(sizes)*len(ratios)
    #初始化锚框数组，anchors=[x_center, y_center, width, height]
    anchors=np.zeros((num_anchors,4))
    #设置锚框的宽度和高度
    anchors[:,2:]=np.tile(sizes,(2,len(ratios))).T
    #调整锚框的宽度和高度（通过乘以ratios[i][0]宽的比例，ratios[i][1]高的比例）
    for i in range(len(ratios)):
        anchors[3*i:3*i+3,2]=anchors[3*i:3*i+3, 2]*ratios[i][0]
        anchors[3*i:3*i+3,3]=anchors[3*i:3*i+3, 3]*ratios[i][1]
    #设置锚框的中心坐标
    anchors[:,0::2]-=np.tile(anchors[:,2]*0.5,(2,1)).T  #(2,1)指的是在行方向上复制2次，在列方向上复制一次
    anchors[:,1::2]-=np.tile(anchors[:,3]*0.5,(2,1)).T
    return anchors
#平移锚框函数：将一组锚框在图像上进行平移，生成一组平移后的锚框。这样可以确保锚框覆盖整个图像
def shift(shape,anchors,stride=config.rpn_stride):
    shift_x=(np.arange(0,shape[0],dtype=keras.backend.floatx())+0.5)*stride  #生成一组x坐标值，用于生成锚框的水平位置
    shift_y=(np.arange(0,shape[1],dtype=keras.backend.floatx())+0.5)*stride  #生成一组y坐标值，用于生成锚框的垂直位置

    shift_x,shift_y=np.meshgrid(shift_x,shift_y) #创建一个网格矩阵，用于生成锚框的位置

    shift_x=np.reshape(shift_x,[-1])
    shift_y=np.reshape(shift_y,[-1]) #将二维数组展平为一维数组

    shifts=np.stack([shift_x,shift_y,shift_x,shift_y],axis=0) #生成一个新的多维数组，原来为(N),现在为(4,N)
    
    shifts=np.transpose(shifts) #转置
    number_of_anchors=np.shape(anchors)[0]
    k=np.shape(shifts)[0]  #获取数组行数，用于确定数组有多少元素

    shifted_anchors=np.reshape(anchors,[1,number_of_anchors,4])+np.array(np.reshape(shifts,[k,1,4]),keras.backend.floatx())  
    shifted_anchors=np.reshape(shifted_anchors,[k*number_of_anchors,4])  #对数组进行形状重塑，改变维度
    return shifted_anchors

#获取最终锚框函数：generate_anchors函数生成一组归一化的锚点坐标
def get_anchors(shape,width,height):
    anchors=generate_anchors()  #生成一组锚点坐标
    network_anchors=shift(shape,anchors)  #根据特征图大小shape和生成的锚点坐标anchors，计算出输入图像上对应的归一化锚点坐标
    #将坐标值进行归一化处理
    network_anchors[:,0]=network_anchors[:,0]/width
    network_anchors[:,1]=network_anchors[:,1]/height
    network_anchors[:,2]=network_anchors[:,2]/width
    network_anchors[:,3]=network_anchors[:,3]/height
    #将坐标值剪裁到0-1范围内
    network_anchors=np.clip(network_anchors,0,1) 
    return network_anchors