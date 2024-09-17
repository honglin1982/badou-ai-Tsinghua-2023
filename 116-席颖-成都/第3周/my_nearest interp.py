import cv2
import numpy as np

def f_zoom(ori_img,dst_img):
    ori_w,ori_h,ori_c = ori_img.shape
    dst_w,dst_h       = dst_img[1],dst_img[0]
    emptyImage        = np.zeros((dst_w,dst_h,ori_c),np.uint8)
    sh                = dst_h / ori_h
    sw                = dst_w / ori_w
    for i in range(dst_w):
        for j in range(dst_h):
            x = int(i/sw+0.5)
            y = int(j/sh+0.5)
            emptyImage[i,j] = ori_img[x,y]
    return emptyImage

ori_img = cv2.imread("lenna.png")
zoom = f_zoom(ori_img,(600,600))
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("ori_img",ori_img)
cv2.waitKey(0)



