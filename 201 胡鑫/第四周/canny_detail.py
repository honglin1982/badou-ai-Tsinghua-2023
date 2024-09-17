import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # 灰度化
    path = '../lenna.png'
    img = plt.imread(path)
    # 当使用plt读取png图片时, 获得的矩阵是[0, 1]之间的值
    if path[-4:] == '.png':
        img = img * 255
    # 均值(灰度化), 对最后一个维度(shape为(x, y, 3))做均值化为(x, y, 1),
    # 相当于灰度化
    img = img.mean(axis=-1)
    h, w = img.shape

    # 高斯滤波
    # 通过手动设置sigma的值求对应的高斯核
    sigma = 0.5
    kernel_size = int(round(6*sigma + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    center = kernel_size // 2
    gaussfilter = np.zeros((kernel_size, kernel_size))
    # 套入公式求解高斯核
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            gaussfilter[i, j] = np.exp(-(x**2 + y**2) / (2*sigma**2)) / (2*np.pi*sigma**2)
    # 高斯核归一化
    gaussfilter = gaussfilter / gaussfilter.sum()
    # 边缘处理以进行高斯滤波
    pad = kernel_size // 2
    img_pad = np.pad(img, ((pad, pad), (pad, pad)), 'constant')
    img_gauss = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            img_gauss[i, j] = np.sum(img_pad[i:i+kernel_size, j:j+kernel_size] * gaussfilter)
    
    plt.figure(num='gauss')
    plt.imshow(img_gauss.astype(np.uint8), cmap='gray')

    # sobel
    # sobel算子相当于是在求导, 即计算梯度
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros((h, w))
    img_tidu_y = np.zeros((h, w))
    img_tidu = np.zeros((h, w))
    # 边缘填补
    img_pad = np.pad(img_gauss, ((1, 1), (1, 1)), 'constant')
    for i in range(h):
        for j in range(w):
            img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_y)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
    plt.figure(num='tidu')
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')

    # 非极大值抑制
    img_yizhi = np.zeros((h, w))
    # 防止运行时报错
    img_tidu_x[img_tidu_x==0] = 0.00000001
    angle = img_tidu_y / img_tidu_x
    for i in range(1, h-1):
        for j in range(1, w-1):
            temp = img_tidu[i-1:i+2, j-1:j+2]
            flag = True
            # 正常坐标系,x向右,y向上,但是坐标值是像素坐标的值(...)
            # 一下的取值均是建立在这个基础上
            # 思想是双线性插值的思想
            if angle[i, j] <= -1:
                # tidu_y的绝对值>tidu_x, 异号, 所以权重为-1/angle
                # num_1 = -temp[0, 0] / angle[i, j] + (1 + 1/angle[i, j]) * temp[0, 1]
                # num_1 = -temp[0, 0] / angle[i, j] + temp[0, 1] + temp[0, 1] / angle[i, j]
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if angle[i, j] >= 1:
                # tidu_y的绝对值>tidu_x, 同号, 所以权重为1/angle
                # num_1 = temp[0, 2] / angle[i, j] + (1-1/angle[i, j])*temp[0, 1]
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if angle[i, j] < 0:
                # tidu_y的绝对值<tidu_x, 异号, 所以权重为-angle
                # num_1 = -temp[0, 0] * angle[i, j] + (1+angle[i, j])*temp[1, 0]
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if angle[i, j] > 0:
                # tidu_y的绝对值<tidu_x, 同号, 所以权重为angle
                # num_1 = temp[0, 2] * angle[i, j] + (1-angle[i, j]) * temp[1, 2]
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(num='yizhi')
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')

    # 双阈值控制
    # 大多数是凭经验和结果选取
    lowerboundary = img_tidu.mean() * 0.5
    highboundary = lowerboundary * 3
    zhan = []
    # 将所有大于高阈值的点定义为强边缘点
    for i in range(h):
        for j in range(w):
            if img_yizhi[i, j] > highboundary:
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] < lowerboundary:
                img_yizhi[i, j] = 0 
    # 遍历每个强边缘点的八个领域, 若存在大于低阈值小于高阈值的若边缘点,
    # 则此若边缘点也认为是边缘点
    while not len(zhan) == 0:
        temp1, temp2 = zhan.pop()
        for i in range(temp1-1, temp1+2):
            for j in range(temp2-1, temp2+2):
                if i != temp1 and j != temp2:
                    if lowerboundary < img_yizhi[i, j] < highboundary:
                        img_yizhi[i, j] = 255
                        zhan.append([i, j])
    # 将非边缘点置为0
    for i in range(h):
        for j in range(w):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0

    plt.figure('boundary')
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()