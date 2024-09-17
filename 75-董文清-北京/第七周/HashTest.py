import numpy as np
import os
import cv2
import time

#均值hash
def aHash(img, w=8,h=8):

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.sum(gray) / (w * h)
    return ''.join(['1' if gray[i, j] > mean else '0' for i in range(h) for j in range(w)])

#差值hash
def dHash(img, w=9,h=8):

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return ''.join(['1' if gray[i, j] > gray[i, j + 1] else '0' for i in range(h) for j in range(h)])
#感知hash
def pHash(img, w = 32, h = 32):

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #dct 容器
    dct = np.zeros((h, w), np.float32)
    dct[:h, :w] = gray
    #dct变换
    dct = cv2.dct(cv2.dct(dct))
    imgList = dct.flatten()
    mean = sum(imgList) / (h * w)
    imgList = ['1' if val > mean else '0' for val in imgList]
    return ''.join('%x'%int(''.join(imgList[x:x+4]), 2) for x in range(0, len(imgList), 4))

def cmpHash(s1, s2):

    return 1 - sum([c1 != c2 for c1, c2 in zip(s1, s2)]) * 1.0 / len(s1)

#精度比较
def hashCmp(img1, img2):

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    timeS = time.time()
    print("均值hash, 原图像与色彩增强后图像相似度为%.2f%%, 耗时%.4f s"
          %(cmpHash(aHash(img1), aHash(img2)) * 100, time.time() - timeS))
    timeS = time.time()
    print("差值hash, 原图像与色彩增强后图像相似度为%.2f%%, 耗时%.4f s"
          %(cmpHash(dHash(img1), dHash(img2)) * 100, time.time() - timeS))
    timeS = time.time()
    print("感知hash, 原图像与色彩增强后图像相似度为%.2f%%, 耗时%.4f s"
          %(cmpHash(pHash(img1), pHash(img2)) * 100, time.time() - timeS))

def aHashTimeCmp(img1, img2, loops=1000):

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    timeS = time.time()
    for _ in range(loops):
        hash1 = aHash(img1)
        hash2 = aHash(img2)
        cmpHash(hash1, hash2)
    print("均值hash, 运行%d次耗时%.4f"%(loops, time.time() - timeS))


def dHashTimeCmp(img1, img2, loops=1000):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    timeS = time.time()
    for _ in range(loops):
        hash1 = dHash(img1)
        hash2 = dHash(img2)
        cmpHash(hash1, hash2)
    print("差值hash, 运行%d次耗时%.4f" % (loops, time.time() - timeS))


def pHashTimeCmp(img1, img2, loops=1000):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    timeS = time.time()
    for _ in range(loops):
        hash1 = pHash(img1)
        hash2 = pHash(img2)
        cmpHash(hash1, hash2)
    print("感知hash, 运行%d次耗时%.4f" % (loops, time.time() - timeS))

#测试
def test(dict=".\source"):

    files = [file for file in os.listdir(dict)]
    base = os.path.join(dict, files[0])
    files = files[1:]
    color = os.path.join(dict, files[1])
    #精度比较
    hashCmp(base, color)
    #耗时比较
    for file in files:

        path = os.path.join(dict, file)
        aHashTimeCmp(base, path)
        dHashTimeCmp(base, path)
        pHashTimeCmp(base, path)



if __name__ == '__main__':

    test();