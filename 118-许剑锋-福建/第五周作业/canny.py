import numpy as np
import cv2
import math

def show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()


def gaussion_kernel(kernel_size = 3, sigma = 1.2):
    kernel = np.zeros((kernel_size, kernel_size))
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[x][y] = np.exp(-1 * (x ** 2 + y ** 2)/ (2 * (sigma ** 2)))
    return kernel


def gaussion_kernel2(kernel_size = 3, sigma = 1.2):
    kernel = np.zeros((kernel_size, kernel_size))
    temp = [i - kernel_size//2 for i in range(kernel_size)]
    n1 = 1/(2 * math.pi*sigma*2)
    n2 = -1/(2*sigma*2)
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[x][y] = n1 * np.exp(n2 * (temp[x]**2 + temp[y] ** 2))
    kernel = kernel / kernel.sum()
    print(kernel.shape)
    return kernel


def canny_detail(gray_img):
    # 高斯平滑
    sigma = 0.5
    kernel_size = 3
    # 计算高斯平滑后的图像

    gauss_kernel = gaussion_kernel(kernel_size, sigma)
    dim = int(np.round(6 * sigma + 1))
    if dim % 2 == 0:
        dim += 1
    # pad_img = np.pad(gray_img, (kernel_size // 2, kernel_size // 2),  'constant', constant_values=0)
    pad_img = np.pad(gray_img, (dim // 2, dim // 2),  'constant', constant_values=0)

    # new_img = gray_img.copy()
    # width, height = gray_img.shape
    # for i in range(width):
    #     for j in range(height):
    #         val = np.mean(pad_img[i:i+kernel_size, j:j+kernel_size] * gauss_kernel)
    #         new_img[i][j] = val

    gauss_filter = gaussion_kernel2(dim, sigma)
    new_img = gray_img.copy()
    width, height = gray_img.shape
    for i in range(width):
        for j in range(height):
            val = np.sum(pad_img[i:i+dim, j:j+dim] * gauss_filter)
            new_img[i][j] = val
    show(new_img,'new_img')

    # 使用sobel算子计算梯度 dx dy
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img_x = np.zeros(gray_img.shape)
    img_y = np.zeros(gray_img.shape)
    img_tidu = np.zeros(gray_img.shape)
    for i in range(width):
        for j in range(height):
            x =  np.sum(pad_img[i:i+kernel_size, j:j+kernel_size] * sobel_kernel_x)
            img_x[i][j] = x if x > 0 else 0.000001
            img_y[i][j] = np.sum(pad_img[i:i+kernel_size, j:j+kernel_size] * sobel_kernel_y)
            img_tidu[i][j] = np.sqrt(img_x[i][j] ** 2 + img_y[i][j] ** 2)

    angle = img_y / img_x

    # 非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            temp = pad_img[i-1:i+2, j-1:j+2]
            print(i,j)
            # 八条边，四分类
            pixel1, pixel2 = 0, 0
            if angle[i][j] >= 1:
                pixel1 = (temp[0][2] - temp[0][1]) / angle[i][j] + temp[0][1]
                pixel2 = (temp[2][1] - temp[2][0]) / angle[i][j] + temp[2][0]
            elif angle[i][j] >= 0:
                pixel1 = (temp[2][2] - temp[1][2]) / angle[i][j] + temp[1][2]
                pixel2 = (temp[1][0] - temp[2][0]) / angle[i][j] + temp[2][0]
            elif angle[i][j] >= -1:
                pixel1 = (temp[0][0] - temp[1][0]) / angle[i][j] + temp[1][0]
                pixel2 = (temp[1][2] - temp[2][2]) / angle[i][j] + temp[2][2]
            else:
                pixel1 = (temp[0][1] - temp[0][0]) / angle[i][j] + temp[0][0]
                pixel2 = (temp[2][2] - temp[2][1]) / angle[i][j] + temp[2][1]
            if img_tidu[i][j] >= pixel1 and img_tidu[i][j] >= pixel2:
                img_yizhi[i][j] = img_tidu[i][j]
    show(img_yizhi, 'img_yizhi')
    # 双阈值检测，连接
    low_boundary = img_tidu.mean() * 0.5
    high_boundary = low_boundary * 3
    stack = [] # 边缘连接
    for i in range(1, img_yizhi.shape[0] - 1):
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i][j] >= high_boundary:
                img_yizhi[i][j] = 255
                stack.append([i, j])
            if img_yizhi[i][j] <= low_boundary:
                img_yizhi[i][j] = 0
    show(img_yizhi, 'img_yizhi_second')

    while len(stack) > 0:
        temp_x, temp_y = stack.pop()
        a = img_yizhi[temp_x-1: temp_x+2, temp_y-1: temp_y+2]
        if a[0][0] < high_boundary and a[0][0] > low_boundary:
            img_yizhi[temp_x-1][temp_y-1] = 255
            stack.append([temp_x-1, temp_y-1])
        if a[0][1] < high_boundary and a[0][1] > low_boundary:
            img_yizhi[temp_x - 1][temp_y] = 255
            stack.append([temp_x - 1, temp_y])
        if a[0][2] < high_boundary and a[0][2] > low_boundary:
            img_yizhi[temp_x - 1][temp_y + 1] = 255
            stack.append([temp_x - 1, temp_y + 1])
        if a[1][0] < high_boundary and a[1][0] > low_boundary:
            img_yizhi[temp_x][temp_y - 1] = 255
            stack.append([temp_x, temp_y - 1])
        if a[1][1] < high_boundary and a[1][1] > low_boundary:
            img_yizhi[temp_x][temp_y] = 255
            stack.append([temp_x, temp_y])
        if a[1][2] < high_boundary and a[1][2] > low_boundary:
            img_yizhi[temp_x][temp_y + 1] = 255
            stack.append([temp_x, temp_y + 1])
        if a[2][0] < high_boundary and a[2][0] > low_boundary:
            img_yizhi[temp_x + 1][temp_y - 1] = 255
            stack.append([temp_x + 1, temp_y - 1])
        if a[2][1] < high_boundary and a[2][1] > low_boundary:
            img_yizhi[temp_x + 1][temp_y] = 255
            stack.append([temp_x + 1, temp_y])
        if a[2][2] < high_boundary and a[2][2] > low_boundary:
            img_yizhi[temp_x + 1][temp_y + 1] = 255
            stack.append([temp_x + 1, temp_y + 1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i][j] != 255:
                img_yizhi[i][j] = 0

    show(img_yizhi, 'img_yizhi_third')



def cv2_canny(gray_img, width, height):
    canny = cv2.Canny(gray_img, width, height)
    show(canny, 'cv2_canny')


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2_canny(gray, 200, 300)
    canny_detail(gray)



