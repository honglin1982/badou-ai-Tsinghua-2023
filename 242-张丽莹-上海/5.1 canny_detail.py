import numpy as np
import matplotlib.pyplot as plt
import math


if __name__ == "main":
    #1.灰度化：用plt模块读图，png格式浮点数从1扩大为255，进行灰度化
    src = "lenna.png"
    img = plt.imread(src)
    if src[-4:] == ".png":
        img = img * 255
    img = img.mean(axis=-1)

    #2.高斯平滑
    #步骤：定sigma、定卷积核大小、存储高斯核、计算高斯核、进行卷积（填充、卷积）、展示结果
    #定sigma
    sigma = 0.5
    # 定卷积核大小
    dim = int(np.round(sigma*6))
    if dim%2 == 0:
        dim += 1
    #高斯核、套公式
    Gaussian_filter = np.zeros([dim,dim])
    tmp = [i-dim//2 for i in range(dim)]
    n1 = 1/(2*math.pi*sigma**2)
    n2 = -1/(2*sigma**2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i,j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    Gaussian_filter = Gaussian_filter/Gaussian_filter.sum()
    #卷积，定一个新图，加填充层，进行卷积
    dx, dy = img.shape
    img_new = np.zeros(img.shape)
    tmp = dim//2
    img_pad = np.pad(img, ((dim, dim), (dim, dim)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)
    #展示结果
    plt.figure(1)
    plt.imshow(img_new.astype(np.unit8), cmap='gray')
    plt.axis('off')

    #3.sobel算子
    #步骤：sobel卷积核、定被卷积的梯度图（x、y、合成图共三张）、填充原图、卷积（for循环）、求角度、展示结果
    sobel_kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)  #存储梯度图像
    img_tidu_y = np.zeros(img_new.shape)
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1,1),(1,1)),'constant')
    dx, dy = img_new.shape
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i,j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)
            img_tidu_y[i,j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)
            img_tidu[i,j] = np.sqrt(img_tidu_x[i,j]**2 + img_tidu_y[i,j]**2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y/img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.unit8),cmap='gray')
    plt.axis('off')

    #4.非极大值抑制：由sobel算子已经做完边缘提取，进行第一轮筛选：每个点梯度与周围的点梯度比较，挑出梯度最大的算作边缘。
    #步骤：定一张抑制后的图，for遍历每个点求其8邻域内的最大值并保留
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx-1):
        for j in range(1,dx-2):
            flag = True
            temp = img_tidu[i-1:i+2, j-1:j+2]
            if angle[i,j] <= -1:
                num1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            elif angle[i,j] >= 1:
                num1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            elif angle[i, j] > 0:
                num1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            elif angle[i, j] < 0:
                num1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
            plt.figure(3)
            plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
            plt.axis('off')

    #5.双阈值检测，连接边缘。遍历nms法求得的所有认为是边的点，检测其是否是孤岛，孤岛划为0，如有连接则认为是边缘，划为255
    #步骤：定义阈值、定一个栈、for遍历img_yizhi所有点 取强弃弱、
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3
    zhan = []    #定义一个强边缘的栈
    for i in range(1, img_yizhi.shape[0]-1):
        for j in range(1, img_yizhi.shape[1]-1):
            if img_yizhi[i, j] >= high_boundary:
                img_yizhi[i, j] = 255
            elif img_yizhi[i, j] <= lower_boundary:
                img_yizhi[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()   #出栈
        a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 -1] = 255
            zhan.append([temp_1 - 1, temp_2 - 1])
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
            if img_yizhi[i, j] !=0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0


    #绘图
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.unit8), cmap='gray')
    plt.axis('off')  #关闭坐标刻度值
    plt.show()
