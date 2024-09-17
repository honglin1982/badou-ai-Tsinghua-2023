import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    # 图片灰度化，采用像素均值
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.png':
        img = img * 255
    img = img.mean(axis=-1)

    # 高斯平滑
    sigm = 0.5  # 设置高斯核参数
    dim = int(np.round(6 * sigm + 1))
    if dim % 2 == 0:
        dim = dim + 1
    gaosihe = np.zeros([dim, dim])  # 高斯核
    tmp = [i - dim // 2 for i in range(dim)]  # 生成序列  -1,0,1
    n1 = 1 / (2 * math.pi * sigm ** 2)
    n2 = -1 / (2 * sigm ** 2)
    for i in range(dim):
        for j in range(dim):
            gaosihe[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    gaosihe = gaosihe / gaosihe.sum()
    dy, dx = img.shape
    img_new = np.zeros(img.shape)
    by = dim // 2  # 边缘
    img_pad = np.pad(img, ((by, by), (by, by)), 'constant')  # 边缘填补
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j: j + dim] * gaosihe)

    # 设置梯度
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_y)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x

    # 非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1,dx -1):
        for j in range(1, dy - 1):
            flag = True
            temp = img_tidu[i-1 :i+2,j-1:j+2]
            if angle[i,j] <= -1:
                num1 = (temp[0,1] - temp[0,0])/ angle[i,j] +temp[0,1]
                num2 = (temp[2,1] - temp[2,2])/ angle[i,j] +temp[2,1]
                if not  (img_tidu[i,j]>num1 and img_tidu[i,j] > num2):
                    flag = False
            elif angle[i,j]>1:
                num1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            elif angle[i,j]>0:
                num1 = (temp[0, 2] - temp[1, 2]) / angle[i, j] + temp[1, 2]
                num2 = (temp[2, 0] - temp[1, 0]) / angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            elif angle[i,j]<0:
                num1 = (temp[1, 0] - temp[0, 0]) / angle[i, j] + temp[1, 0]
                num2 = (temp[1, 2] - temp[2, 2]) / angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            if flag:
                img_yizhi[i,j] = img_tidu[i,j]

    # 双阈值检测
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
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

    plt.figure('高斯平滑')
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.figure('原图')
    plt.imshow(img, cmap='gray')
    plt.figure('梯度')
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.figure('非极大值抑制')
    plt.imshow(img_yizhi.astype(np.uint8),cmap='gray')
    plt.show()
