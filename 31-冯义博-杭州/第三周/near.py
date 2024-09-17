import cv2
import numpy as np


# 邻近插值法
def near_inter(source_img, rh, rw):
    h, w, c = source_img.shape
    n_img = np.zeros((rh, rw, c), np.uint8)
    hr = rh / h
    wr = rw / w
    for i in range(rh):
        for j in range(rw):
            y = int(i / hr + 0.5)
            if y > h:
                y = h - 1
            x = int(j / wr + 0.5)
            if x > w:
                x = w - 1
            n_img[i, j] = source_img[y, x]
    return n_img


# 双线性插值法
def bilinear_img(s_img, dh, dw):
    h, w, c = s_img.shape
    n_img = np.zeros((dh, dw, c), dtype=np.uint8)
    scale_x, scale_y = float(w) / dw, float(h) / dh
    for i in range(3):
        for x in range(dw):
            for y in range(dh):
                src_x = (x + 0.5) * scale_x - 0.5
                src_y = (y + 0.5) * scale_y - 0.5
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, h - 1)

                temp0 = (src_x1 - src_x) * s_img[src_y0, src_x0, i] + (src_x - src_x0) * s_img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * s_img[src_y1, src_x0, i] + (src_x - src_x0) * s_img[src_y1, src_x1, i]
                n_img[y, x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return n_img


img = cv2.imread("../lenna.png")
cv2.imshow("source", img)
cv2.imshow("near", near_inter(img, 600, 700))
cv2.imshow("bilinear", bilinear_img(img, 700, 700))
cv2.waitKey(0)
