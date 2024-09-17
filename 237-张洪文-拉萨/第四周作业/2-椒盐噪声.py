import cv2
import numpy as np

"""
给一副数字图像加上椒盐噪声的处理顺序: 
    1.指定信噪比 SNR（信号和噪声所占比例） ，其取值范围在[0, 1]之间
    2.计算总像素数目 SP， 得到要加噪的像素数目 NP = SP * SNR
    3.随机获取要加噪的每个像素位置P（i, j）
    4.指定像素值为255或者0。
    5.重复3, 4两个步骤完成所有NP个像素的加噪
"""

# 接受 原始图像 与 信噪比 参数
def impulse_noise(img, snr):
    # 深拷贝原始图像
    noise_img = img.copy()

    # 计算要加噪的像素数量
    noise_number = int(img.size * snr)

    # 随机获取要加噪的每个像素位置P
    # 1、得到每个像素的一维索引值数组
    noise_positions = np.random.choice(img.size, size=noise_number, replace=False)
    # 2、根据一维索引得到多维索引位置，如果为3通道图像，该数组有三个子数组，分别对应 h，w，c
    noise_index = np.unravel_index(indices=noise_positions, shape=img.shape)
    # 3、根据多维数组位置并进行判断赋值
    noise_img[noise_index] = np.where(np.random.uniform(size=noise_number) < 0.5, 0, 255)

    return noise_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png", 0)  # 灰度模式
    # img = cv2.imread("lenna.png")  # 默认彩色

    # 指定信噪比为 0.2
    noise_img = impulse_noise(img, snr=0.2)

    cv2.imshow("source_img", img)
    cv2.imshow("dst_img", noise_img)
    cv2.waitKey(0)