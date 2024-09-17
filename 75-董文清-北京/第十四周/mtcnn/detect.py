import  cv2
import  numpy as np
from mtcnn import  mtcnn

img = cv2.imread('img/timg.jpg')
model = mtcnn()
threshold = [0.3, 0.5, 0.8] #三段网络置信度阈值
rectangles = model.detectFace(img, threshold)
draw = img.copy()
for rectangle in rectangles:
    if rectangle is not None:
        W = int(rectangle[2]) - int(rectangle[0])
        H = int(rectangle[3]) - int(rectangle[1])
        paddingH = 0.01 * W
        paddingW = 0.02 * H
        cropImg = img[int(rectangle[1] + paddingH) : int(rectangle[3] - paddingH),
                      int(rectangle[0] - paddingW) : int(rectangle[2] + paddingW)]
        if cropImg is None : continue
        if cropImg.shape[0] < 0 or cropImg.shape[1] < 0: continue
        cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])),
                      (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)
        for i in range(5, 15, 2): cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])),
                                             2, (0, 255, 0))

cv2.imwrite('img/out.jpg', draw)
cv2.imshow('test', draw)
c = cv2.waitKey(0)