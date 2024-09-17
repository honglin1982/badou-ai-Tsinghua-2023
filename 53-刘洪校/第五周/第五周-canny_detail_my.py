# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:50:53 2023

@author: lhx
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
 
if __name__ == '__main__':
    pic_path = 'lenna.png' 
    img = plt.imread(pic_path)
    # pic_path[-4:]  截取后四位
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img = img.mean(axis=-1)  # 取均值就是灰度化了，平均值算法
                             # axis=-1表示原始张量维度减1，比如（512，512，3）变成（512，512）此处也可写成axis=2
    #1.高斯平滑
    # 高斯核：
    # 重要的是需要理解，高斯卷积核大小的选择将影响Canny检测器的性能：
    # 尺寸越大，检测器对噪声的敏感度越低，但是边缘检测的定位误差也将略有增加。
    # 一般5x5是一个比较不错的trade off
    # ===================cv2高斯平滑  开始===================
    # 参数解释
    # src:输入图像
    # ksize:(核的宽度,核的高度)，输入高斯核的尺寸，核的宽高都必须是正奇数。否则，将会从参数sigma中计算得到。
    # sigmaX:高斯核在X方向上的标准差。
    # sigmaY:高斯核在Y方向上的标准差。默认为None，如果sigmaY=0，则它将被设置为与sigmaX相等的值。如果这两者都为0,则它们的值会从ksize中计算得到。计算公式为：
    # borderType:像素外推法，默认为None
    dst=cv2.GaussianBlur(img, (5,5), 0.5)
    plt.figure("cv2高斯平滑")
    plt.imshow(dst.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')#关闭坐标轴
    # ===================cv2高斯平滑  结束===================
    
    # ===================手动高斯平滑  开始===================
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    dim = int(np.round(6 * sigma + 1))  # round是四舍五入函数，根据标准差求高斯核是几乘几的，也就是维度
    if dim % 2 == 0:  # 最好是奇数,不是的话加一
        dim += 1
        
    # 第一步：构建高斯矩阵  
    Gauss_filter = np.zeros([dim, dim])  # 高斯滤波器，存储高斯核，这是数组不是列表了
    #Gauss_filter = np.zeros([5, 5])
    
    # 得到中心点的位置
    cH = (dim - 1) / 2
    cW = (dim - 1) / 2
    # 计算gauss(sigma, r, c)
    for x in range(dim):
        for y in range(dim):
            #math.pow(x, y) 方法返回返回 x 的 y 次幂
            norm2 = math.pow(x - cH, 2) + math.pow(y - cH, 2)
            #等同于norm2 = (x - cH)**2 + (y - cH)**2
            #math.exp(x) 方法返回 e 的 x 次幂（次方）Ex，其中 e = 2.718281... 是自然对数的基数
            Gauss_filter[x][y] = math.exp(-norm2 / (2 * math.pow(sigma, 2)))
            #Gauss_filter[x][y] = 1/(2*math.pi*math.pow(sigma, 2))*math.exp(-norm2 / (2 * math.pow(sigma, 2)))
    #print(Gauss_filter)
    # 第二步：计算高斯矩阵的和
    sumGM = np.sum(Gauss_filter)
    #print(sumGM)
    # 第三步：归一化 #因为最后要归一化，所以在代码实现中可以去掉高斯函数中的系数1/2Π*sigma^2。高斯卷积算子翻转180度和本身是相同的。
    Gauss_filter = Gauss_filter / sumGM
    #print(Gauss_filter)
    dx, dy = img.shape
    img_new = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    tmp = dim//2
    #print(img)
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补
    #print(img_pad)
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gauss_filter)
    plt.figure("手动高斯平滑")
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')

    
    # ===================手动高斯平滑  结束===================
    
    # 2、求梯度。以下两个是滤波求梯度用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，卷积核是3，所以添加一层就行，根据上面矩阵结构所以写1
    for i in range(dx):
        for j in range(dy):
            # img_pad[i:i+3, j:j+3] 从左到右，从上到下，依次取卷积核大小的矩阵
            img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)  # x方向滑动
            img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)  # y方向滑动
            # np.sqrt 开方  a平方+b平方=c平方
            img_tidu[i, j] = np.sqrt( math.pow(img_tidu_x[i, j],2) +  math.pow(img_tidu_y[i, j],2))
    img_tidu_x[img_tidu_x == 0] = 0.00000001#除数不能为0，把矩阵中0换成0.00000001
    angle = img_tidu_y/img_tidu_x
    plt.figure("梯度")
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')
    
    
    # 3、非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    # 边缘一圈排除 从内圈开始
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = True  # 在8邻域内是否要抹去做个标记
            # 取该点为中心的3*3矩阵，9个点
            temp = img_tidu[i-1:i+2, j-1:j+2]  # 梯度幅值的8邻域矩阵
            # 使用线性插值法判断抑制与否
            if angle[i, j] <= -1:  #取左上角和上，右下角和下，四个点
                #num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                #num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False#比其他两个大，不需要抑制
            elif angle[i, j] >= 1:#取右上角和上，左下角和下，四个点
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:#取右上角和右，左下角和左，四个点
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:#取左上角和左，右下角和右，四个点
                #num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                #num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure("需要抑制的")
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    
    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_yizhi.shape[0]-1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1]-1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0
 
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1-1, temp_2-1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1-1, temp_2-1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])
 
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
 
    # 绘图
    plt.figure("边缘结果")
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()