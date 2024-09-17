#cannpy边缘检测算法
# 1.对图像进行灰度
# 2.对图像进行高斯滤波
#  2.1实现用py生成高斯卷积核
#  2.2实现卷积

# 3.检测图像中的水平垂直对角边缘（如prewitt，sobe算子等）
#     3.1两次sobe算子进行卷积操作一次左 一次下

# 4对梯度值进行非极大值抑制
# 5.用双阈值算法检测和链接边缘

import cv2
import numpy as np
from sympy import true


# 生成高斯卷积核 size为大小 ，sigma为标准差，标准差越大，滤波效果越明显，平滑程度越高；标准差越小，滤波效果越细致，平滑程度越低
def generate_gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2
    total = 0

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            exponent = -(x**2 + y**2) / (2 * sigma**2)
            kernel[i, j] = np.exp(exponent)
            total += kernel[i, j]

    kernel /= total

    return kernel
# 生成sobel算子
def generate_sobel_kernel(size, axis):
    # Define Sobel kernels
    if size == 3:
        if axis == 'x':
            kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        elif axis == 'y':
            kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif size == 5:
        if axis == 'x':
            kernel = np.array(
                [[-1, -2, 0, 2, 1], [-2, -4, 0, 4, 2], [-4, -8, 0, 8, 4], [-2, -4, 0, 4, 2], [-1, -2, 0, 2, 1]])
        elif axis == 'y':
            kernel = np.array(
                [[-1, -2, -4, -2, -1], [-2, -4, -8, -4, -2], [0, 0, 0, 0, 0], [2, 4, 8, 4, 2], [1, 2, 4, 2, 1]])
    else:
        raise ValueError("Invalid kernel size. Only 3x3 and 5x5 Sobel kernels are supported.")

    return kernel

# 卷积操作
def apply_gaussian_filter(image, kernel):
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 获取卷积核的大小
    kernel_size = kernel.shape[0]

    # 计算需要在图像边缘填充的像素数
    padding = kernel_size // 2

    # 为图像进行边缘填充
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)

    # 创建一个与原始图像大小相同的空白图像用于存储卷积结果
    filtered_image = np.zeros_like(image)
    filtered_image = np.zeros(image.shape)
    # 对图像进行卷积
    for i in range(height):
        for j in range(width):
            # 获取卷积区域
            region = padded_image[i:i+kernel_size, j:j+kernel_size]

            # 对卷积区域与卷积核进行元素相乘并求和
            filtered_value = np.sum(region * kernel)
            # if filtered_value < 0:
            #     filtered_value = 0
            # 将卷积结果存储到输出图像中
            filtered_image[i, j] = filtered_value



    return filtered_image
# 计算梯度值输出两张图一张梯度值，一张边缘检测
def combine_images(image1, image2):

    # 将两张图片的灰度值平方和开根号得到新的图片
    combined_image = np.sqrt(image1**2 + image2**2).astype(np.uint8)

    image1[image1 == 0] = 0.00000001
    angle_image = image2.astype(np.float32) / image1.astype(np.float32)

    return combined_image,angle_image
# 用线性插值计算局部极大值
def bitwise_and(img_tidu,angle):
    dx ,dy = img_tidu.shape
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                if abs(angle[i, j]) < 1e-6:
                    angle[i, j] = 1e-6
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                if abs(angle[i, j]) > 1e+6:
                    angle[i, j] = 1e+6
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    return img_yizhi

def threshold(img,low,hight):
    imgpoint=[]
    dx, dy = img.shape
    #遍历所有点，中间态放入
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            if img[i,j]<hight and img[i,j]>low:
                imgpoint.append([i,j])
            elif img[i,j]>hight:
                img[i, j] =255
            elif img[i, j] < low:
                img[i, j] = 0
    while len(imgpoint)>0:
        i ,j = imgpoint.pop()
        # 8个临点判断
        if img[i-1,j+1]>hight:
            img[i,j]=255
            # imgpoint.append([i-1,j+1])
        if img[i,j+1]>hight:
            img[i,j]=255
            # imgpoint.append([i,j+1])
        if img[i+1,j+1]>hight:
            img[i,j]=255
            # imgpoint.append([i+1,j+1])
        if img[i-1,j]>hight:
            img[i,j]=255
            # imgpoint.append([i-1,j])
        if img[i+1,j]>hight:
            img[i,j]=255
            # imgpoint.append([i+1,j])
        if img[i-1,j-1]>hight:
            img[i,j]=255
            # imgpoint.append([i-1,j-1])
        if img[i,j-1]>hight:
            img[i,j]=255
            # imgpoint.append([i,j-1])
        if img[i+1,j-1]>hight:
            img[i,j]=255
            # imgpoint.append([i+1,j-1])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] != 0 and img[i, j] != 255:
                img[i, j] = 0

    return  img

def canny_edge_detection(image, threshold1, threshold2):
    # 1. 对图像进行灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 对图像进行高斯滤波
    blurred = apply_gaussian_filter(gray,generate_gaussian_kernel(3,0.5))

    # 3. sobel算子卷积
    sobel_x = apply_gaussian_filter(blurred,generate_sobel_kernel(3,"x"))
    sobel_y = apply_gaussian_filter(blurred,generate_sobel_kernel(3,"y"))

    # 计算梯度值
    sobel_z ,angle_image = combine_images(sobel_x,sobel_y)
    # 4. 对梯度值进行非极大值抑制
    img_yizhi= bitwise_and(sobel_z,angle_image)
    # # 5. 用双阈值算法检测和链接边
    edges2 = threshold(img_yizhi, 50, 150, )

    return edges2

# 读取图像
image = cv2.imread('lenna.png')

# 设定阈值
threshold1 = 100
threshold2 = 210

# 进行边缘检测
edges = canny_edge_detection(image, threshold1, threshold2)

# 显示结果
cv2.imshow('原图', image)
cv2.imshow('灰度化之后', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 保存放大后的图像
output_path = '灰度化后.jpg'
cv2.imwrite(output_path, edges)