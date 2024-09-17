import cv2
import numpy as np
import math

# 手写最临近插值
def zoomImg(rawImg, zoom2Height, zoom2Width):
    height, width, channels = rawImg.shape # 512, 512, 3
    targetImg = np.zeros((zoom2Height, zoom2Width, channels), np.uint8) # uint8的范围：0-255，可表示灰度范围

    sh = zoom2Height / height
    sw = zoom2Width / width
    print(height)
    print(width)
    print(channels)

    for i in range(zoom2Height):
        for j in range(zoom2Width):
            x = math.floor(i / sh + 0.5)
            y = math.floor(j / sw + 0.5)
            if x < height and y < width:
                targetImg[i, j] = rawImg[x, y]
    return targetImg


img = cv2.imread('./lenna.png')
zoom = zoomImg(img, 1200, 1200)
print(zoom)
print(zoom.shape)

cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0) # 不加这一句图片闪一下就关上了
