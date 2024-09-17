'''
Canny边缘检测算法，手写
TODO: 效果和cv2封装好的有差异，需要有时间再回来检查一下
'''
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv

if __name__ == '__main__':
    pic_path = 'lenna.png'

    # -------------- step1. 灰度化 ----------------
    '''
        matplotlib.pyplot.imread：读取一张图片，将图像数据变成数组array,
        这里读取RGB图像，返回三维数组,
        png图像以浮点数组(0-1)的形式返回
    '''
    img = plt.imread(pic_path) # ，这里读取RGB图像，返回三维数组
    # print(img)
    img = img * 255
    img = img.mean(axis = -1) # 取平均值灰度化

    # --------------step2. 高斯平滑 --------------------
    # TODO: 手写太复杂了，以后有时间了再回来写
    img_new = cv.GaussianBlur(img, [5, 5], 0)
    plt.figure('高斯滤波')
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')
    # plt.show()


    # --------------step3. Sobel检测边缘、求梯度 --------------------
    dx, dy = img.shape
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_grad_x = np.zeros(img_new.shape)
    img_grad_y = np.zeros([dx, dy])
    img_grad = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补1
    for i in range(dx):
        for j in range(dy):
            img_grad_x[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_x) # x方向
            img_grad_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_y) # y方向
            img_grad[i, j] = np.sqrt(img_grad_x[i, j]**2 + img_grad_y[i, j]**2)
    img_grad_x[img_grad_x == 0] = 0.00000001
    tan = img_grad_y/img_grad_x
    plt.figure('Sobel边缘检测')
    plt.imshow(img_grad.astype(np.uint8), cmap='gray')
    plt.axis('off')
    # plt.show()


    # --------------step4. 非极大值抑制 --------------------
    img_yizhi = np.zeros(img_grad.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = True  # 在8邻域内是否要抹去做个标记，True=需要抹去
            temp = img_grad[i-1:i+2, j-1:j+2] # 梯度幅值的8邻域矩阵
            # 使用线性插值法判断抑制与否
            if tan[i, j] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / tan[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / tan[i, j] + temp[2, 1]
                if not (img_grad[i, j] > num_1 and img_grad[i, j] > num_2):
                    flag = False
            elif tan[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / tan[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / tan[i, j] + temp[2, 1]
                if not (img_grad[i, j] > num_1 and img_grad[i, j] > num_2):
                    flag = False
            elif tan[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * tan[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * tan[i, j] + temp[1, 0]
                if not (img_grad[i, j] > num_1 and img_grad[i, j] > num_2):
                    flag = False
            elif tan[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * tan[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * tan[i, j] + temp[1, 2]
                if not (img_grad[i, j] > num_1 and img_grad[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_grad[i, j]
    plt.figure('非极大值抑制二值图')
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    # plt.show()


    # --------------step5. 双阈值检测 --------------------
    lower_boundary = img_grad.mean() * 0.5
    high_boundary = lower_boundary * 3
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
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
            img_yizhi[temp_1-1, temp_2-1] = 255 # 这个像素点标记为边缘
            zhan.append([temp_1-1, temp_2-1]) # 进栈
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
    plt.figure('双阈值检测后最终效果')
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()


    # 直接调用封装好的Canny算法：
    img = cv.imread("lenna.png", 1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("canny", cv.Canny(gray, 200, 300))
    cv.waitKey()
    cv.destroyAllWindows()

