import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

def cannyDetail(src):

    #灰度化
    img = plt.imread(src)
    if src[-4:] == '.png': img *= 255 #由于png格式存储的是浮点数 所以要×255
    img = img.mean(axis=-1)           #按行取均值实现灰度化

    #高斯滤波
    #求出高斯算子
    sigma = 0.5
    dim = int(np.round(sigma * 6 + 1))
    if dim % 2 == 0: dim += 1   #尽量保证算子是奇数*奇数
    GaussianFilter = np.zeros([dim, dim])
    seq = [i - dim //2 for i in range(dim)]
    div1 = 1/(2 * math.pi * sigma **2)
    div2 = -1/(2 * sigma **2)
    for i in range(dim):
        for j in range(dim):
            GaussianFilter[i, j] = div1 * math.exp(div2 * (seq[i] ** 2 + seq[j] ** 2))
    GaussianFilter = GaussianFilter / GaussianFilter.sum()

    #进行高斯滤波
    dx, dy = img.shape
    imgNew = np.zeros(img.shape)
    padLen = dim // 2
    imgPad = np.pad(img, ((padLen, padLen),(padLen, padLen)), 'constant') #添加两圈填充
    for i in range(dx):
        for j in range(dy):
            imgNew[i, j] = np.sum(imgPad[i:i+dim, j:j+dim] * GaussianFilter)
    plt.figure(1)
    plt.imshow(imgNew.astype(np.uint8), cmap='gray')
    plt.axis('off')

    #用Sobel算子求梯度
    sobelX = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
    sobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    imgGradX = np.zeros(imgNew.shape)
    imgGradY = np.zeros(imgNew.shape)
    imgGrad = np.zeros(imgNew.shape)
    imgPad = np.pad(imgNew, ((1,1),(1,1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            imgGradX[i, j] = np.sum(sobelX * imgPad[i:i+3, j:j+3])
            imgGradY[i, j] = np.sum(sobelY * imgPad[i:i+3, j:j+3])
            imgGrad[i, j] = np.sqrt(imgGradX[i, j] ** 2 + imgGradY[i, j] ** 2)

    plt.figure(2)
    plt.imshow(imgGrad.astype(np.uint8), cmap='gray')
    plt.axis('off')

    #非极大值抑制
    imgGradX[imgGradX == 0] = 0.00000000001 #处理分母为0的情况
    imgRestrain = np.zeros(imgGrad.shape)
    angle = imgGradY / imgGradX
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            zone = imgGrad[i-1:i+2, j-1:j+2]
            flag = True
            if angle[i, j] <= -1:

                pos = (zone[0, 1] - zone[0, 0]) / angle[i, j] + zone[0, 1]
                neg = (zone[2, 1] - zone[2, 2]) / angle[i, j] + zone[2, 1]
                if not (imgGrad[i, j] > pos and imgGrad[i, j] > neg) : flag = False
            elif angle[i, j] >= 1:

                pos = (zone[0, 2] - zone[0, 1]) / angle[i, j] + zone[0, 1]
                neg = (zone[2, 0] - zone[2, 1]) / angle[i, j] + zone[2, 1]
                if not (imgGrad[i, j] > pos and imgGrad[i, j] > neg): flag = False
            elif angle[i, j] > 0:

                pos = (zone[0, 2] - zone[1, 2]) * angle[i, j] + zone[1, 2]
                neg = (zone[2, 0] - zone[1, 0]) * angle[i, j] + zone[1, 0]
                if not (imgGrad[i, j] > pos and imgGrad[i, j] > neg): flag = False

            elif angle[i, j] < 0:

                pos = (zone[1, 0] - zone[0, 0]) * angle[i, j] + zone[1, 0]
                neg = (zone[1, 2] - zone[2, 2]) * angle[i, j] + zone[1, 2]
                if not (imgGrad[i, j] > pos and imgGrad[i, j] > neg): flag = False
            if flag : imgRestrain[i, j] = imgGrad[i, j]
    plt.figure(3)
    plt.imshow(imgRestrain.astype(np.uint8), cmap='gray')
    plt.axis('off')

    #双阈值检测
    lowBoundary = np.mean(imgGrad) * 0.5
    highBoundary = lowBoundary * 3
    stack = []
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            if imgRestrain[i, j] >= highBoundary:
                imgRestrain[i, j] = 255
                stack.append([i, j])
            elif imgRestrain[i, j] <= lowBoundary:
                imgRestrain[i, j] = 0

    while len(stack) != 0 :

        i, j = stack.pop()
        zone = imgRestrain[i - 1 : i + 2, j - 1 : j + 2]
        if zone[0, 0] > lowBoundary and zone[0, 0] < highBoundary:
            imgRestrain[i - 1, j - 1] = 255
            stack.append([i - 1, j - 1])
        if zone[0, 1] > lowBoundary and zone[0, 1] < highBoundary:
            imgRestrain[i - 1, j] = 255
            stack.append([i - 1, j])
        if zone[0, 2] > lowBoundary and zone[0, 2] < highBoundary:
            imgRestrain[i - 1, j + 1] = 255
            stack.append([i - 1, j + 1])
        if zone[1, 0] > lowBoundary and zone[1, 0] < highBoundary:
            imgRestrain[i, j - 1] = 255
            stack.append([i, j - 1])
        if zone[1, 2] > lowBoundary and zone[1, 2] < highBoundary:
            imgRestrain[i, j + 1] = 255
            stack.append([i, j + 1])
        if zone[2, 0] > lowBoundary and zone[2, 0] < highBoundary:
            imgRestrain[i + 1, j - 1] = 255
            stack.append([i + 1, j - 1])
        if zone[2, 1] > lowBoundary and zone[2, 1] < highBoundary:
            imgRestrain[i + 1, j] = 255
            stack.append([i + 1, j])
        if zone[2, 2] > lowBoundary and zone[2, 2] < highBoundary:
            imgRestrain[i + 1, j + 1] = 255
            stack.append([i + 1, j + 1])

    for i in range(dx):
        for j in range(dy):imgRestrain[i, j] = 0 if imgRestrain[i, j] != 255 else 255
    plt.figure(4)
    plt.imshow(imgRestrain.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()

def sobelLaplaceCanny(src):

    img = cv2.imread(src, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    laplace = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    canny = cv2.Canny(gray, 10, 300)
    plt.subplot(231),plt.imshow(gray, cmap='gray'),plt.title("Original")
    plt.subplot(232),plt.imshow(sobelX, cmap='gray'),plt.title("SobelX")
    plt.subplot(233),plt.imshow(sobelY, cmap='gray'),plt.title("SobelY")
    plt.subplot(234),plt.imshow(laplace, cmap='gray'),plt.title("Laplace")
    plt.subplot(235),plt.imshow(canny, cmap='gray'),plt.title("Canny")
    plt.show()


def cannyTrack(lowThreshold):

    detectedEdges = cv2.GaussianBlur(gray, (5, 5), 0)
    detectedEdges = cv2.Canny(detectedEdges, lowThreshold, lowThreshold*ratio, apertureSize=kernel)
    dst = cv2.bitwise_and(img, img, mask=detectedEdges)
    cv2.imshow('canny Demo', dst)

if __name__ == '__main__':

    picPath = 'lenna.png'
    cannyDetail(picPath)
    sobelLaplaceCanny(picPath)
    lowThreshold = 0
    maxThreshold = 100
    ratio = 3
    kernel = 3
    img = cv2.imread(picPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('canny Demo')
    cv2.createTrackbar('Threshold', 'canny Demo', lowThreshold, maxThreshold,cannyTrack)
    cannyTrack(0)
    if cv2.waitKey(0) == 27: cv2.destroyAllWindows()
