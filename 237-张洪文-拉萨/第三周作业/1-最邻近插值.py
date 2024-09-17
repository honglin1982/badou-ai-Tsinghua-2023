import cv2
import numpy as np

"""
最邻近插值（Nearest Neighbor Interpolation）是一种图像插值方法，用于在图像放大或缩小时估计新像素的值。
最邻近插值的原理：对于目标图像中的每个新像素位置，找到源图像中最接近的像素位置，并将该像素的值赋给目标像素。

对于每个目标像素位置（x', y'），找到源图像中最接近的像素位置（x0, y0）。
放大操作：x0 = round(x)，y0 = round(y)。  四舍五入
缩小操作：x0 = floor(x)，y0 = floor(y)。  向下取整

在放大操作中，使用四舍五入将目标像素位置映射到最接近的源图像像素位置，以获取更准确的插值结果。
而在缩小操作中，通过向下取整来选择最接近的源图像像素位置，因为在缩小过程中会有像素丢失，因此向下取整可以更好地保留图像细节。
"""

# 接收一个图像高宽的元组，用于放大或缩小图像
def Nearest_Interpolation(img, image_size):
    height, width, channels = img.shape
    print("源图像大小:", height, width, channels)

    empty_image = np.zeros((image_size[0], image_size[1], channels), img.dtype)
    print("目标图像大小:", empty_image.shape)

    # 得出缩放比例
    h_ratio = image_size[0]/height
    w_ratio = image_size[1]/width
    print(f"高度的缩放比例: {h_ratio}, 宽度的缩放比例: {w_ratio}")

    for i in range(image_size[0]):
        for j in range(image_size[1]):
            # 使用round()  进行四舍五入操作
            # 这里的 i/sh, j/sw 是按照比例得出放大后的图像其对应的原始图像的像素坐标值
            if h_ratio > 1:
                x = round(i/h_ratio)
            else:
                x = int(np.floor(i/h_ratio))
            if w_ratio > 1:
                y = round(j/w_ratio)
            else:
                y = int(np.floor(j/w_ratio))
            print(x, y)
            empty_image[i, j] = img[x, y]

    cv2.imshow("nearest interp", empty_image)
    cv2.imshow("image", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    img = cv2.imread("lenna.png")  # 指定缩放的图像
    size = (128, 128)  # 指定图像放大或缩小后的大小
    Nearest_Interpolation(img, size)


