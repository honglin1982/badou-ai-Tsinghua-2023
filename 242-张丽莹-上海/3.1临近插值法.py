import cv2
import numpy as np


def function(img, out_dim):
    h, w, c = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    dst_img = np.zeros((dst_h, dst_w, 3), np.uint8)
    sc_h = float(dst_h)/h
    sc_w = float(dst_w)/w
    for i in range(dst_h):
        for j in range(dst_w):
            x = int(i/sc_w + 0.5)
            y = int(i/sc_h + 0.5)
            dst_img[i, j] = img[x, y]
    return dst_img


if __name__ == '__main__':
    img1 = cv2.imread("lenna.png")
    zoom = function(img1, (800, 800))
    print(zoom)
    print(zoom.shape)
    cv2.imshow("nearest interp", zoom)
    cv2.imshow("image", img1)
    cv2.waitKey(0)
