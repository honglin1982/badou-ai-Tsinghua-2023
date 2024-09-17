import numpy as np
import cv2


# 双线性插值
def interp_bilinear(img, h_t, w_t):
    h_o, w_o, c = img.shape
    if h_t == h_o and w_t == w_o:
        return img.copy()
    img_t = np.zeros([h_t, w_t, c], dtype=img.dtype)
    scale_x, scale_y = float(w_o) / w_t, float(h_o) / h_t   # 计算缩放比例
    for n in range(c):
        for i in range(h_t):
            y = (i + 0.5) * scale_y - 0.5   # 几何中心对称后，找出目标像素点在原图坐标上对应的height
            # 在原图坐标纵轴上找出相邻的4个点对应的上下两个height值
            y1 = int(np.floor(y))
            y2 = min(y1 + 1, h_o - 1)
            for j in range(w_t):
                x = (j + 0.5) * scale_x - 0.5   # 几何中心对称后，找出目标像素点在原图坐标上对应的width
                # 在原图坐标横轴上找出相邻的4个点对应的左右两个width值,与纵轴两个height值一起获取到4个相邻点的像素值
                x1 = int(np.floor(x))
                x2 = min(x1 + 1, w_o - 1)

                # 将4个相邻点的像素值代入公式，计算插入点的像素值
                fx1 = (x2 - x) * img[y1, x1, n] + (x - x1) * img[y1, x2, n]
                fx2 = (x2 - x) * img[y2, x1, n] + (x - x1) * img[y2, x2, n]
                img_t[i, j, n] = int((y2 - y) * fx1 + (y - y1) * fx2)

    return img_t


if __name__ == "__main__":
    img0 = cv2.imread("lenna.png")
    cv2.imshow("Original", img0)
    cv2.waitKey()
    img1 = interp_bilinear(img0, 900, 900)
    cv2.imshow("Target", img1)
    cv2.waitKey()
    # 调用resize方法实现，参数interpolation的默认值是INTER_LINEAR，可省略
    img2 = cv2.resize(img0, (900, 900), interpolation=cv2.INTER_LINEAR)  # 这里尺寸顺序是width, height
    cv2.imshow("Target", img2)
    cv2.waitKey()
