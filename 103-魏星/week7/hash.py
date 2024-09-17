import cv2
import numpy as np

'''
均值哈希算法
1.缩放:图片缩放为8*8
2.灰度化:转换为灰度图
3.求平均值:计算灰度图所有像素的平均值
4.比较:像素值大于平均值记作1,相反记作0,总共64位
5.生成hash
6.对比指纹,计算汉明距离
'''
def avg_hash(img):
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    avg = np.average(img_gray)

    hash_str = ''
    for i in range(8):
        for j in range(8):
            if img_gray[i,j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

'''
差值哈希
1.缩放:图片缩放为8*9
2.灰度化:转换为灰度图
4.比较:像素值大于后一个像素值记作1,相反记作0.本行不与下一行对比,每行9个像素,8个差值,有8行,总共64位
5.生成hash
6.对比指纹,计算汉明距离
'''
def diff_hash(img):
    img = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    avg = np.average(img_gray)

    hash_str = ''
    for i in range(8):
        for j in range(8):
            if img_gray[i,j] > img_gray[i,j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

def getHanMingDistance(hashStr1,hashStr2):
    distance = 0
    for i in range(64):
        if hashStr1[i] != hashStr2[i]:
            distance += 1
    return distance

def cmpHash(img1,img2):
    avgHashStr1 = avg_hash(img1)
    avgHashStr2 = avg_hash(img2)
    print("均值Hash-img1:", avgHashStr1)
    print("均值Hash-img2:", avgHashStr2)
    print("均值Hash比较结果:",getHanMingDistance(avgHashStr1,avgHashStr2))

    diffHashStr1 = diff_hash(img1)
    diffHashStr2 = diff_hash(img2)
    print("差值Hash-img1:", diffHashStr1)
    print("差值Hash-img2:", diffHashStr2)
    print("差值Hash比较结果:", getHanMingDistance(diffHashStr1, diffHashStr2))


img1 = cv2.imread("flower.png")
img2 = cv2.imread("flower_noise.png")
cmpHash(img1,img2)