import cv2
import numpy as np

"""
1. 缩放：图片缩放为8*8，保留结构，除去细节。
2. 灰度化：转换为灰度图。
3. 求平均值：计算灰度图所有像素的平均值。
4. 比较：像素值大于平均值记作1，相反记作0，总共64位。
5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
6. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样的，不
相同位数越少，图片越相似。
"""
# 均值哈希:返回哈希值
def mean_hash(img, width=8, height=8):
    # 1 图像缩放为（8,8），立法插值法(cv2.INTER_CUBIC),使用相邻的16个像素的加权平均值来填充新像素
    src_img = cv2.resize(img, dsize=(8, 8), interpolation=cv2.INTER_CUBIC)

    # 2 灰度化
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    # 3 求平均像素值
    pixel_mean = np.mean(gray_img)
    print("图像平均值:", pixel_mean)

    # 4 比较并生成hash
    img_hash = ""
    for i in gray_img.flatten():  # 展平为1位数组
        if i > pixel_mean:
            img_hash += "1"
        else:
            img_hash += "0"

    print(f"图像哈希为:{img_hash}")
    return img_hash


if __name__ == '__main__':
    image = cv2.imread("image/lenna.png")
    mean_hash(img=image)
