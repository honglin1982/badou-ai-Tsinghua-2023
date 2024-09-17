import cv2
import numpy as np

def bilinear_interp(ori_img,dst_img):
    ori_w,ori_h,ori_c = ori_img.shape
    dst_w,dst_h       = dst_img[1],dst_img[0]
    print("ori_w,ori_h =",ori_w,ori_h)
    print("dst_w,dst_h =",dst_w,dst_h)
    if ori_w == dst_w and ori_h == dst_h:
        return ori_img.copy()
    emptyImage = np.zeros((dst_w,dst_h,3),dtype=np.uint8)
    scale_x, scale_y = float(dst_w) / ori_w, float(dst_h) / ori_h
    for i in range(3):
        for dst_x in range(dst_w):
            for dst_y in range(dst_h):
                #中心对称
                ori_x = (dst_x + 0.5) / scale_x - 0.5
                ori_y = (dst_y + 0.5) / scale_y - 0.5
                # 找到将用于计算插值的点的坐标
                ori_x0 = int(np.floor(ori_x))
                ori_x1 = min(ori_x0+1 ,ori_w-1)
                ori_y0 = int(np.floor(ori_y))
                ori_y1 = min(ori_y0+1,ori_h-1)
                #计算插值
                temp0 = (ori_x1-ori_x) * ori_img[ori_x0,ori_y0,i] + (ori_x - ori_x0)*ori_img[ori_x1,ori_y0,i]
                temp1 = (ori_x1-ori_x) * ori_img[ori_x0,ori_y1,i] + (ori_x - ori_x0)*ori_img[ori_x1,ori_y1,i]
                emptyImage[dst_x,dst_y,i] = int((ori_y1 -ori_y)*temp0 + (ori_y - ori_y0)*temp1)
    return emptyImage
if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    print(img)
    dst = bilinear_interp(img,(600,600))
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey()




