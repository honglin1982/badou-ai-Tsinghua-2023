import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


# 一、高斯平滑: 接受原始图像和标准差
def Gaussian_smoothing(src_img, sigma):
    # 1、确定高斯核的大小
    dim = int(np.round(6 * sigma + 1))
    if dim % 2 == 0:
        dim += 1
    # 2、根据高斯核大小创建滤波器
    gaussian_filter = np.zeros([dim, dim])
    # 3、生成位置偏移序列：dim//2 的结果是高斯核的半径
    offset = [i - (dim // 2) for i in range(dim)]
    # 4、n1是高斯核的前置系数，用于归一化高斯核的值。n2是高斯核指数部分的系数，用于计算高斯核每个位置的权重
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    # 5、遍历高斯核的每个位置，根据公式计算并存储每个位置的权重值
    for i in range(dim):
        for j in range(dim):
            gaussian_filter[i, j] = n1 * math.exp(n2 * (offset[i] ** 2 + offset[j] ** 2))
    # 6、将高斯核进行归一化处理，使其所有元素的和等于1
    gaus_filter_1 = gaussian_filter / gaussian_filter.sum()
    # 7、创建一个与原图像大小相同的零数组 img_new，用于存储平滑后的图像。
    dx, dy = src_img.shape
    img_dst = np.zeros(src_img.shape)
    # 8、边缘填充：通过 tmp=dim//2 计算边缘填补的大小，利用 np.pad() 函数对原图像进行边缘填补。
    fill_width = dim // 2
    img_pad = np.pad(src_img, fill_width, "constant", constant_values=0)
    # 9、对图像进行高斯平滑处理。循环遍历每个像素 (i, j)，利用高斯核对图像进行加权平均计算
    for i in range(dx):
        for j in range(dy):
            img_dst[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * gaus_filter_1)
    # 10、显示图像
    plt.figure(1)
    plt.imshow(img_dst.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.title('Gaussian Image')
    plt.axis('off')

    return img_dst


# 二、使用Sobel算子计算图像的梯度，包括水平方向和垂直方向的梯度以及梯度的幅值。
def grads(img):
    # 1 使用Sobel算子定义Sobel卷积核：
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # 2 创建存储梯度图像的数组：
    img_grad_x = np.zeros(img.shape)  # 存储水平方向的梯度图像
    img_grad_y = np.zeros(img.shape)  # 存储垂直方向的梯度图像
    img_grad = np.zeros(img.shape)  # 存储梯度幅值图像
    # 3 边缘填补原始图像：3//2 = 1（卷积核大小为3）
    img_pad = np.pad(img, 1, "constant", constant_values=0)
    # 4 遍历图像的每个像素，计算水平方向和垂直方向的梯度与梯度幅值：
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):  # i+3 是因为sobel算子大小为3
            img_grad_x[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_x)
            img_grad_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_y)
            # 计算梯度的幅值：计算梯度在x和y方向上的平方和的平方根来获得梯度的幅值
            img_grad[i, j] = np.sqrt(img_grad_x[i, j]**2 + img_grad_y[i, j]**2)

    # 5 处理梯度方向的值为0的情况，避免除零错误:
    img_grad_x[img_grad_x == 0] = 0.00000001

    # 6 求梯度方向: 梯度方向可以通过计算y方向梯度和x方向梯度的比值来获得。
    angle = img_grad_y / img_grad_x

    # 7 展示梯度幅值图像
    plt.figure(2)
    plt.imshow(img_grad.astype(np.uint8), cmap="gray")
    plt.title('Sobal Image')
    plt.axis("off")

    return img_grad, angle  # 返回梯度幅值和梯度方向


# 三、非极大值抑制
def Non_maximum_suppression(img_amp, angle, d_size):
    # 1、创建存储抑制图像的数组
    img_restrain = np.zeros(img_amp.shape)
    dx, dy = d_size

    # 2、遍历图像内部像素，判断是否抑制
    for i in range(1, dx - 1):  # 循环的起始值和终止值被设置为 (1, dx-1) 和 (1, dy-1)，确保只遍历图像内部的像素。
        for j in range(1, dy - 1):   # 避免处理图像边缘像素时出现索引越界的问题。
            # 以 (i, j) 为中心的 3x3 邻域矩阵
            temp = img_amp[i-1: i+2, j-1: j+2]  # 梯度幅值的8邻域矩阵
            flag = True
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_amp[i, j] > num_1 and img_amp[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_amp[i, j] > num_1 and img_amp[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_amp[i, j] > num_1 and img_amp[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_amp[i, j] > num_1 and img_amp[i, j] > num_2):
                    flag = False
            if flag:
                img_restrain[i, j] = img_amp[i, j]
    # 3、显示图像
    plt.figure(3)
    plt.imshow(img_restrain.astype(np.uint8), cmap='gray')
    plt.title('Non_maximum_suppression Image')
    plt.axis('off')

    return img_restrain


# 四、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
def Dual_threshold_edge(img_gradient, img_restrain):
    # 1、阈值设定：低阈值为梯度图像均值的一半，高阈值是低阈值的三倍
    lower_boundary = img_gradient.mean() * 0.5
    high_boundary = lower_boundary * 3
    # 2、初始化栈
    inn = []
    # 3、判断边缘，遍历像素点，并忽略外圈。
    for i in range(1, img_restrain.shape[0] - 1):
        for j in range(1, img_restrain.shape[1] - 1):
            if img_restrain[i, j] >= high_boundary:  # >=高阈值设强边缘点
                img_restrain[i, j] = 255  # 强边缘点
                inn.append([i, j])   # 入栈,判断8邻域是否有弱边缘
            elif img_restrain[i, j] <= lower_boundary:
                img_restrain[i, j] = 0  # 非边缘点置0
    """
    到目前为止，被划分为强边缘的像素点已经被确定为边缘，因为它们是从图像中的真实边缘中提取出来的。
    然而，对于弱边缘像素，将会有一些争论，因为这些像素可以从真实边缘提取也可以是因噪声或颜
    色变化引起的。为了获得准确的结果，应该抑制由后者引起的弱边缘：
    • 通常，由真实边缘引起的弱边缘像素将连接到强边缘像素，而噪声响应未连接。
    • 为了跟踪边缘连接，通过查看弱边缘像素及其8个邻域像素，只要其中一个为强边缘像素，
    则该弱边缘点就可以保留为真实的边缘。
    这里使用的反向方法：根据强边缘去推断出真实的弱边缘
    """
    # 4、边缘连接：栈不为空，则检查栈中的像素点的8邻域，
    while inn:
        temp_1, temp_2 = inn.pop()  # 出栈
        # 获取弱边缘点的8邻域
        marginal_matrix = img_restrain[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        # offsets 列表存储了8邻域的偏移量
        offsets = [(i, j) for i in range(3) for j in range(3) if not (i == 1 and j == 1)]
        # 遍历每一个邻域位置
        for offset in offsets:
            i, j = offset
            if lower_boundary < marginal_matrix[i, j] < high_boundary:  # 弱边缘
                if img_restrain[temp_1 + i - 1, temp_2 + j - 1] != 255:  # 先判断该像素点是否已被标记，如果未被标记则标记入栈，检查边缘
                    img_restrain[temp_1 + i - 1, temp_2 + j - 1] = 255  # 将这个像素点的实际位置标记为边缘
                    inn.append([temp_1 + i - 1, temp_2 + j - 1])  # 入栈，并继续判断该像素点的8邻域是否存在弱边缘，循环判断，直至没有bi

    # 5、边缘修剪：遍历图像的所有像素点，将非边缘点（像素值不为0和255的点）设为背景点（像素值设为0）
    img_restrain[(img_restrain != 0) & (img_restrain != 255)] = 0

    # 绘制结果图像
    plt.figure(4)
    plt.imshow(img_restrain.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.title("Dual threshold Image")
    plt.show()


if __name__ == '__main__':
    image = cv2.imread("lenna.png", 0)   # 灰度模式
    d_size = image.shape
    # 传入原始图像获得高斯平滑后的图像
    img_smoothness = Gaussian_smoothing(src_img=image, sigma=0.5)
    # 传入平滑图像获得梯度幅值图像和梯度方向
    img_gradient, angle = grads(img_smoothness)
    # 传入梯度幅值与方向和大小获得抑制后的图像
    img_restrain = Non_maximum_suppression(img_gradient, angle, d_size)
    # 传入抑制图像，获得双阈值检测和边缘连接后的图像
    Dual_threshold_edge(img_gradient, img_restrain)
