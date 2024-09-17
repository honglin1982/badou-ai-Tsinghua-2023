import numpy as np
import cv2

def aHash(img, width=8, high=8):
    '''
    均值哈希
    '''
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hash_str = ''

    avg = np.sum(gray) / (width*high)

    for i in range(high):
        for j in range(width):
            if gray[i, j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

def dHash(img, width=9, high=8):
    '''
    差值哈希
    '''
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hash_str = ''
    for i in range(high):
        for j in range(high):
            if gray[i, j] > gray[i, j+1]:
                hash_str += '1'
            else:
                hash_str += '0'

    return hash_str

def cmpHash(hash1, hash2):
    '''
    哈希值相似度比较
    '''
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] == hash2[i]:
            n += 1
    
    return n/len(hash1)

if __name__ == "__main__":
    img1 = cv2.imread('../lenna.png')
    img2 = cv2.imread('../lenna_poisson.png')
    # 均值哈希
    print('均值哈希结果：')
    hash1 = aHash(img1)
    hash2 = aHash(img2)
    print('hash1: ', hash1)
    print('hash2: ', hash2)
    n = cmpHash(hash1, hash2)
    print('均值哈希的相似度为：', n)

    # 差值哈希
    print('差值哈希结果：')
    hash1 = dHash(img1)
    hash2 = dHash(img2)
    print('hash1: ', hash1)
    print('hash2: ', hash2)
    n = cmpHash(hash1, hash2)
    print('差值哈希的相似度为：', n)