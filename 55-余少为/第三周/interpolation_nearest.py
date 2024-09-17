import cv2
import numpy as np


# 最邻近插值
def interp_nearest(img, h_t, w_t):
    h_o, w_o, c = img.shape
    if h_t == h_o and w_t == w_o:
        return img.copy()
    img_t = np.zeros([h_t, w_t, c], dtype=img.dtype)
    sh = h_t / h_o
    sw = w_t / w_o
    for n in range(c):
        for i in range(h_t):
            x = int(i / sh + 0.5)
            x = x-1 if x == h_o else x  # 容错性操作
            for j in range(w_t):
                y = int(j / sw + 0.5)
                y = y-1 if y == w_o else y  # 容错性操作

                img_t[i, j, n] = img[x, y, n]

    return img_t


if __name__ == "__main__":
    img0 = cv2.imread("lenna.png")
    cv2.imshow("Original", img0)
    cv2.waitKey()
    img1 = interp_nearest(img0, 900, 900)
    # print(img1.shape)
    cv2.imshow("Target", img1)
    cv2.waitKey()
    # 调用resize方法实现
    img2 = cv2.resize(img0, (900, 900), interpolation=cv2.INTER_NEAREST)    # 这里尺寸顺序是width, height
    cv2.imshow("Target", img2)
    cv2.waitKey()
