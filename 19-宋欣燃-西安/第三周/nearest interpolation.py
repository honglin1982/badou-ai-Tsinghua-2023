import cv2
import numpy as np

def function(img,out_dim):
    height, width, channels = img.shape #512,512,3
    emptyImage = np.zeros((out_dim[0], out_dim[1], channels), np.uint8)
    sh = out_dim[0]/height
    sw = out_dim[1]/width
    for i in range(out_dim[0]):
        for j in range(out_dim[1]):
            x = int(i/sh+0.5)
            y = int(j/sw+0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage

if __name__ == '__main__':
    img = cv2.imread("/Users/songxinran/Documents/GitHub/badou-AI-Tsinghua-2023/lenna.png")
    zoom = function(img, (700, 700))
    print(img.shape)
    print(zoom)
    print(zoom.shape)
    cv2.imshow("nearest interp", zoom)
    cv2.imshow("img", img)
    cv2.waitKey()