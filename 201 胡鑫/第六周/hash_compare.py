import cv2
import numpy as np
import os
import time

def aHash(img, width=8, high=8):
    """均值哈希

    Args:
        img (ndarray): 图像矩阵
        width (int, optional): 缩放宽度. Defaults to 8.
        high (int, optional): 缩放高度. Defaults to 8.
    """
    # 缩放成8*8，使用三次插值法
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 初始化hash
    hash_str = ''
    
    # 像素均值
    avg = np.sum(gray) / (width*high)
    
    # 将每个点与avg比较，若大于则为1，否则为0，生成hash值
    for i in range(high):
        for j in range(width):
            if gray[i, j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

def dHash(img, width=9, high=8):
    """插值哈希

    Args:
        img (ndarray): 图像矩阵
        width (int, optional): 缩放宽度. Defaults to 9.
        high (int, optional): 缩放高度. Defaults to 8.
    """
    # 缩放成8*9，使用三次插值法
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 初始化哈希值
    hash_str = ''

    # 同一行，前一个点与后一个点比较，若大于为1，否则为0，生成hash值
    for i in range(high):
        for j in range(high):
            if gray[i, j] > gray[i, j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

def pHash(img, width=64, high=64):
    """感知哈希

    Args:
        img (ndarray): 图像矩阵
        width (int, optional): 缩放宽度. Defaults to 64.
        high (int, optional): 缩放高度. Defaults to 64.
    """
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    # 灰度化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 准备
    h, w = img.shape[:2]
    vis0 = np.zeros( (h, w), np.float32 )
    vis0[:h, :w] = img

    # 二维dct变换，并缩放成（32，32）
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize((32, 32))

    # 转换成一维
    img_list = vis1.flatten()

    # 计算均值
    avg = np.sum(img_list) / len(img_list)
    avg_list = ['0' if i > avg else '1' for i in img_list]

    hash_str = ''
    # 对于 avg_list 中每个 4 个元素为一组的子列表，进行如下操作
    for x in range(0, 32 * 32, 4):
        # 将当前子列表中的 4 个二进制字符串拼接成一个新的二进制串，并将其转换为对应的十六进制字符
        hex_char = '%x' % int(''.join(avg_list[x:x + 4]), 2)
        # 将当前十六进制字符添加到结果字符串中
        hash_str += hex_char
    return hash_str

def cmp_hash(hash1, hash2):
    """hash值对比

    Args:
        hash1 (str): 第一个hash
        hash2 (str): 第二个hash
    """
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        # 不相等则n计数+1，n为不相似度
        if hash1[i] != hash2[i]:
            n += 1
    # 返回最终相似度
    return 1 - n / len(hash1)

def test_diff_hash(img_path1, img_path2, loops=1000):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    start_time = time.time()

    for _ in range(loops):
        hash1 = dHash(img1)
        hash2 = dHash(img2)
        cmp_hash(hash1, hash2)

    print(">>> 执行%s次耗费的时间为%.4f s." % (loops, time.time() - start_time))

def test_avg_hash(img_path1, img_path2, loops=1000):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    start_time = time.time()

    for _ in range(loops):
        hash1 = aHash(img1)
        hash2 = aHash(img2)
        cmp_hash(hash1, hash2)

    print(">>> 执行%s次耗费的时间为%.4f s." % (loops, time.time() - start_time))

def test_p_hash(img_path1, img_path2, loops=100):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    start_time = time.time()

    for _ in range(loops):
        hash1 = pHash(img1)
        hash2 = pHash(img2)
        cmp_hash(hash1, hash2)

    print(">>> 执行%s次耗费的时间为%.4f s." % (loops, time.time() - start_time))

def test_aHash(img1, img2):
    hash1 = aHash(img1)
    hash2 = aHash(img2)
    info = cmp_hash(hash1, hash2)
    return f'ahash计算图像的最终相似度为{info}\n'

def test_dHash(img1, img2):
    hash1 = dHash(img1)
    hash2 = dHash(img2)
    info = cmp_hash(hash1, hash2)
    return f'dhash计算图像的最终相似度为{info}\n'

def test_pHash(img1, img2):
    hash1 = pHash(img1)
    hash2 = pHash(img2)
    info = cmp_hash(hash1, hash2)
    return f'phash计算图像的最终相似度为{info}\n'

def deal(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    info = ''
    # 计算图像hash相似度
    info += test_aHash(img1, img2)
    info += test_dHash(img1, img2)
    info += test_pHash(img1, img2)
    return info

def contact_path(file_name):
    output_path = './source'
    return os.path.join(output_path, file_name)

def main():
    data_img_name = 'lenna.png'
    name = data_img_name.split(".")[0]
    
    # 图片地址
    base = contact_path(data_img_name)
    light = contact_path('%s_light.jpg' % name)
    resize = contact_path("%s_resize.jpg" % name)
    constrast = contact_path("%s_constrast.jpg" % name)
    sharp = contact_path("%s_sharp.jpg" % name)
    blur = contact_path("%s_blur.jpg" % name)
    color = contact_path("%s_color.jpg" % name)
    rotate1 = contact_path("%s_rotate1.jpg" % name)

    # 测试效率
    print('dHash效率：')
    test_diff_hash(base, base)
    test_diff_hash(base, light)
    test_diff_hash(base, resize)
    test_diff_hash(base, constrast)
    test_diff_hash(base, sharp)
    test_diff_hash(base, blur)
    test_diff_hash(base, color)
    test_diff_hash(base, rotate1)

    print('dHash效率：')
    test_avg_hash(base, base)
    test_avg_hash(base, light)
    test_avg_hash(base, resize)
    test_avg_hash(base, constrast)
    test_avg_hash(base, sharp)
    test_avg_hash(base, blur)
    test_avg_hash(base, color)
    test_avg_hash(base, rotate1)

    print('pHash效率：')
    test_p_hash(base, base)
    test_p_hash(base, light)
    test_p_hash(base, resize)
    test_p_hash(base, constrast)
    test_p_hash(base, sharp)
    test_p_hash(base, blur)
    test_p_hash(base, color)
    test_p_hash(base, rotate1)

    # 测试算法精度，base与light
    print('精度：')
    info = deal(base, light)
    print(info)

if __name__ == "__main__":
    main()

