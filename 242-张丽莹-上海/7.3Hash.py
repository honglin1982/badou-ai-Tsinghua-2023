import cv2
import numpy as np
import time

# 均值哈希算法 averange hashing
def aHash(img, width=8, height=8):
    """
    均值哈希算法
    param img:图像数据
    param width:图像数据缩放后宽度
    param height:图像数据缩放后高度

    return:均值哈希序列
    """

    # 获取图像数据，缩放、灰度化
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC) # 哈希算法里resize插值方法用三次样条插值
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #求像素值均值
    sum_pixel = 0
    for i in range(8):
        for j in range(8):
            sum_pixel = sum_pixel +gray[i, j]

    mean_pixel = sum_pixel / 64

    #求均值哈希序列，像素值大于均值时为1，小于均值时为0
    hash_str = ""
    for i in range(8):
        for j in range(8):
            if gray[i, j] > mean_pixel:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str



# 差值哈希算法 difference hashing
def dHash(img, width=9, height=8):
    """
        差值哈希算法
        param img:图像数据
        param width:图像数据缩放后宽度, 注意差值哈希的宽度额外加1，是9
        param height:图像数据缩放后高度

        return:差值哈希序列
        """

    # 获取图像数据，缩放、灰度化
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 求差值哈希序列，共8行，每行的像素值大于后一个时为1
    hash_str = ""
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 感知哈希算法 perceptual hashing
def pHash(img, width=64, height=64):
    """
    感知哈希算法
    param img:图像数据
    param width:图像数据缩放后宽度
    param height:图像数据缩放后高度

    return:感知哈希序列
    """
    # 获取灰度图，缩放图
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)  # 哈希算法里resize插值方法用三次样条插值
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建二维列表
    h, w = gray.shape[:2]
    vis0 = np.zeros((h, w), np.float32)   # 创建全0矩阵
    vis0[:h, :w] = gray   # 将gray复制到vis0的左上角区域。因大小相同，完成重叠

    # 二维DCT变换
    vis1 = cv2.dct(cv2.dct(vis0))  # 先对矩阵vis0进行dct变换得到变换矩阵，对它再进行dct变换得到结果
    vis1.resize(32, 32)   # 为什么是32？

    # 将二维list降维到一维list
    img_list = vis1.flatten()

    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i > avg else '1' for i in img_list]

    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x+4]), 2) for x in range(0, 32*32, 4)])


# 哈希值比较函数
def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


img1 = cv2.imread('mountain1.png')
img2 = cv2.imread('mountain2.png')

ahash1 = aHash(img1)
ahash2 = aHash(img2)
print(ahash1)
print(ahash2)
n1=cmpHash(ahash1,ahash2)
print('均值哈希算法相似度：',n1)

dhash1 = dHash(img1)
dhash2 = dHash(img2)
print(dhash1)
print(dhash2)
n2=cmpHash(dhash1,dhash2)
print('差值哈希算法相似度：',n2)

phash1 = pHash(img1)
phash2 = pHash(img2)
print(phash1)
print(phash2)
n3=cmpHash(phash1,phash2)
print('感知哈希算法相似度：',n3)





"""
扩展——感知哈希算法原理及步骤：
    1.缩小图像尺寸
    将原始图像缩小到固定的尺寸，如8x8或32x32
    这样可以减少计算量，并且能够去除图像中的细节信息，只保留图像的主要特征。
    
    2.转换为灰度图像
    将缩小后的图像转换为灰度图像，这样可以降低图像处理的复杂度，使得计算更加简单。
    
    3.计算DCT变换
    对灰度图像进行DCT（离散余弦变换）变换，得到图像的频域信息。
    由于人眼对低频信息比较敏感，因此可以只保留DCT变换后的前N个系数，其他系数舍去。
    
    4.计算平均值
    计算DCT系数的平均值，得到一个阈值，将DCT系数中大于平均值的记为1，小于平均值的记为0。
    这样就得到了一个二进制哈希值。
    
    5.得到最终哈希值
    根据二进制哈希值中的1和0的分布，得到最终的哈希值。
    一般来说，可以将二进制哈希值中的排列顺序随机打乱，这样可以增加哈希值的鲁棒性，防止受到图像旋转、裁剪等操作的影响。
"""

"""
扩展——cv2.resize函数中插值方法：
    cv2.INTER_NEAREST：最近邻插值
    cv2.INTER_LINEAR：双线性插值（默认）
    cv2.INTER_AREA：区域插值
    cv2.INTER_CUBIC：三次样条插值
    cv2.INTER_LANCZOS4：Lanczos 滤波器插值
    哈希算法里用的是三次样条插值，优势是它使用了更高阶的多项式进行逼近，因此能够更好地处理曲线和曲面上的变化。
    同时，三次样条插值能够通过边界条件的设定，更好地控制插值结果在边界处的行为，避免出现不连续或者震荡的情况。
"""
