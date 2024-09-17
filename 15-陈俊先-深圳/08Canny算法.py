import numpy as np
import matplotlib.pyplot as plt
import math
import pylab

if __name__ == '__main__':
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.png':
        img = img * 255
    img = img.mean(axis=-1)  # 取均值即灰度化

    # 高斯平滑（目的是使边缘检测避免噪声干扰）
    sigma = 0.5
    dim = int(np.round(6 * sigma + 1))
    if dim % 2 == 0:
        dim += 1
    Gaussian_filter = np.zeros([dim, dim])  # 储存高斯核
    tmp = [i - dim//2 for i in range(dim)]
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    dx, dy = img.shape
    img_new = np.zeros(img.shape)  # 储存平滑之后的图像，zeros函数得到的是浮点型数据
    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
    '''
    表示对图像进行边缘填充操作，对于二维图像
    (temp, temp)表示在垂直方向上在顶部和底部填充 temp 行
    (temp, temp) 表示在水平方向上在左侧和右侧填充 temp 列
    填充的值使用 'constant' 模式，即在边缘填充固定的常数值
    '''
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
            # img_pad[i: i + dim, j: j + dim] 选取以 (i, j) 为起始位置，大小为 dim 的子图像块
    plt.figure(1)   # plt.figure(1) 用于创建一个新的图形窗口，编号为 1
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')  # 用于关闭坐标轴显示的功能
    plt.show()

    # sobel边缘检测
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_grad_x = img_grad_y = img_grad = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_grad_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            img_grad_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_grad[i, j] = np.sqrt(img_grad_x[i, j] ** 2 + img_grad_y[i, j] ** 2)
    img_grad_x[img_grad_x == 0] = 0.00000001  # 为了避免在后续计算中出现除以 0 的错误
    angle = img_grad_y / img_grad_x
    plt.figure(2)
    plt.imshow(img_grad.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()

    # 非极大值抑制
    img_suppression = np.zeros(img_grad.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True
            temp = img_grad[i-1:i+2, j-1:j+2]
            if angle[i, j] <= -1:
                num1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_grad[i, j] > num1 and img_grad[i, j] > num2):
                    flag = False
            elif angle[i, j] >= 1:
                num1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_grad[i, j] > num1 and img_grad[i, j] > num2):
                    flag = False
            elif angle[i, j] < 0:
                num1 = (temp[1, 0] - temp[0, 0]) / angle[i, j] + temp[1, 0]
                num2 = (temp[1, 2] - temp[2, 2]) / angle[i, j] + temp[1, 2]
                if not (img_grad[i, j] > num1 and img_grad[i, j] > num2):
                    flag = False
            elif angle[i, j] > 0:
                num1 = (temp[0, 2] - temp[1, 2]) / angle[i, j] + temp[1, 2]
                num2 = (temp[2, 0] - temp[1, 0]) / angle[i, j] + temp[1, 0]
                if not (img_grad[i, j] > num1 and img_grad[i, j] > num2):
                    flag = False
            if flag:
                img_suppression[i, j] = img_grad[i, j]
    plt.figure(5)
    plt.imshow(img_suppression.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()

    # 双阈值检测，边缘链接
    lower_boundary = img_grad.mean() * 0.5
    high_boundary = lower_boundary * 3
    stack = []
    for i in range(1, img_suppression.shape[0] - 1):
        for j in range(1, img_suppression.shape[1] - 1):
            if img_suppression[i, j] >= high_boundary:
                img_suppression[i, j] = 255
                stack.append([i, j])
            elif img_suppression[i, j] <= lower_boundary:
                img_suppression[i, j] = 0

    while not len(stack) == 0:
        temp1, temp2 = stack.pop()
        a = img_suppression[temp1 - 1:temp1 + 2, temp2 - 1:temp2 + 2]   # 以temp1,temp2为中心的8邻域，切片操作
        # 假设 temp_1 的值为 5，那么 temp_1-1 的值为 4，temp_1+2 的值为 7。所以，索引范围 temp_1-1:temp_1+2 表示从索引 4 到索引 6，共计三个点（4、5、6）
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_suppression[temp1 - 1, temp2 - 1] = 255
            stack.append([temp1 - 1, temp2 - 1])
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_suppression[temp1 - 1, temp2] = 255
            stack.append([temp1 - 1, temp2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_suppression[temp1 - 1, temp2 + 1] = 255
            stack.append([temp1 - 1, temp2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_suppression[temp1, temp2 - 1] = 255
            stack.append([temp1, temp2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_suppression[temp1, temp2 + 1] = 255
            stack.append([temp1, temp2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_suppression[temp1 + 1, temp2 - 1] = 255
            stack.append([temp1 + 1, temp2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_suppression[temp1 + 1, temp2] = 255
            stack.append([temp1 + 1, temp2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_suppression[temp1 + 1, temp2 + 1] = 255
            stack.append([temp1 + 1, temp2 + 1])

    for i in range(img_suppression.shape[0]):
        for j in range(img_suppression.shape[1]):
            if img_suppression[i, j] != 0 and img_suppression[i, j] != 255:
                img_suppression[i, j] = 0

    # 绘图
    plt.figure(4)
    plt.imshow(img_suppression.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    pylab.show()
