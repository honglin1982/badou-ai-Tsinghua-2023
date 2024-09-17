import numpy as np
import cv2


def smooth(img_in, sigma, dim):  # 高斯平滑
    # 生成高斯核
    gaussian_filter = np.zeros([dim, dim])
    for i in range(dim):
        for j in range(dim):
            x = i - dim // 2
            y = j - dim // 2
            gaussian_filter[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_filter /= (2 * np.pi * sigma**2)
    gaussian_filter = gaussian_filter / np.sum(gaussian_filter)
    # print('gaussian_filter:\n', gaussian_filter)

    # 生成目标图
    w, h = img_in.shape
    img_out = np.zeros(img_in.shape)
    tmp = dim // 2      # 计算补零数量，接下来进行边缘补零
    img_pad = np.pad(img_in, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘补零
    for i in range(w):
        for j in range(h):
            img_out[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * gaussian_filter)

    return img_out.astype(np.uint8)


def get_gradient_and_angle(img_in):  # 获取梯度和角度
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradients = np.zeros(img_in.shape)
    angle = np.zeros(img_in.shape)

    img_pad = np.pad(img_in, ((1, 1), (1, 1)), 'constant')  # 边缘补零
    for i in range(img_in.shape[0]):
        for j in range(img_in.shape[1]):
            dx = np.sum(img_pad[i:i+3, j:j+3] * Gx)
            dy = np.sum(img_pad[i:i+3, j:j+3] * Gy)
            gradients[i, j] = np.sqrt(dx**2 + dy**2)
            if dx == 0:
                dx = 0.00000001    # 保证除数不为0
            angle[i, j] = dy / dx

    return gradients.astype(np.uint8), angle


def NMS(gradients, angle):
    w, h = gradients.shape
    img_out = np.zeros([w, h])  # 先创建一个全零矩阵

    for i in range(1, w-1):
        for j in range(1, h-1):
            tmp = gradients[i-1:i+2, j-1:j+2]   # 梯度值的八邻域矩阵
            # 进行线性插值，找到该梯度正反向对应点的梯度值
            if angle[i, j] <= -1:
                p1 = (tmp[0, 1] - tmp[0, 0]) / angle[i, j] + tmp[0, 1]
                p2 = (tmp[2, 1] - tmp[2, 2]) / angle[i, j] + tmp[2, 1]
            elif angle[i, j] >= 1:
                p1 = (tmp[0, 2] - tmp[0, 1]) / angle[i, j] + tmp[0, 1]
                p2 = (tmp[2, 0] - tmp[2, 1]) / angle[i, j] + tmp[2, 1]
            elif angle[i, j] > 0:
                p1 = (tmp[0, 2] - tmp[1, 2]) * angle[i, j] + tmp[1, 2]
                p2 = (tmp[2, 0] - tmp[1, 0]) * angle[i, j] + tmp[1, 0]
            elif angle[i, j] < 0:
                p1 = (tmp[1, 0] - tmp[0, 0]) * angle[i, j] + tmp[1, 0]
                p2 = (tmp[1, 2] - tmp[2, 2]) * angle[i, j] + tmp[1, 2]
            # 该梯度值与找到的两个点的梯度值比较，判断该梯度值是否最大
            if gradients[i, j] > p1 and gradients[i, j] > p2:
                img_out[i, j] = gradients[i, j]

    return img_out.astype(np.uint8)


def double_threshold(nms, thrs_min, thrs_max):
    zhan = []   # 创建一个栈，用来存放强边缘点，方便后续检测相邻点
    for i in range(1, nms.shape[0]-1):  # 最外一圈不处理
        for j in range(1, nms.shape[1]-1):
            if nms[i, j] >= thrs_max:   # 大于高阈值，即为强边缘
                nms[i, j] = 255
                zhan.append([i, j])
            elif nms[i, j] < thrs_min:  # 小于低阈值，不是边缘
                nms[i, j] = 0

    while not len(zhan) == 0:
        px, py = zhan.pop()     # 出栈
        tmp = nms[px-1:px+2, py-1:py+2]     # 开始检测八个相邻点
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                if thrs_max > tmp[i, j] >= thrs_min:
                    nms[px+i-1, py+j-1] = 255
                    zhan.append([px+i-1, py+j-1])

    for i in range(nms.shape[0]):
        for j in range(nms.shape[1]):
            if nms[i, j] != 0 and nms[i, j] != 255:
                nms[i, j] = 0

    return nms


# 1、图片灰度化
img = cv2.imread('lenna.png', 0)
# cv2.imshow('gray', img)
# cv2.waitKey()

# 2、高斯平滑
img_smoothed = smooth(img, 1.2, 5)
# cv2.imshow('smoothed', np.hstack([img, img_smoothed]))
# cv2.waitKey()

# 3、利用sobel算子求梯度和角度
img_grad, arr_angle = get_gradient_and_angle(img_smoothed)
# print(img_grad.mean()*0.5)
# cv2.imshow('Gradient', img_grad)
# cv2.waitKey()

# 4、非极大值抑制
img_nms = NMS(img_grad, arr_angle)
cv2.imshow('Gradient & NMS', np.hstack([img_grad, img_nms]))
cv2.waitKey()

# 5、双阈值检测
img_dbthrsd = double_threshold(img_nms.copy(), 32, 80)
cv2.imshow('DoubleThreshold(20 60)', np.hstack([img_nms, img_dbthrsd]))
cv2.waitKey()
