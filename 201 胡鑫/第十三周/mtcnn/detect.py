import cv2
import numpy as np
from mtcnn import mtcnn

img = cv2.imread('./img/timg.jpg')
model = mtcnn()
threshold = [.5, .6, .7]  # 三段网络的置信度阈值不同
rectangles = model.detectFace(img, threshold)
draw = img.copy()

for rectangle in rectangles:
    if rectangle is not None:
        w = int(rectangle[2]) - int(rectangle[0])
        h = int(rectangle[3]) - int(rectangle[1])
        padding_h = .01 * w
        padding_w = .02 * h

        crop_img = img[int(rectangle[1] + padding_h):int(rectangle[3] - padding_h),
                   int(rectangle[0] - padding_w):int(rectangle[2] + padding_w)]

        if crop_img is None:
            continue
        if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
            continue
        cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                      (255, 0, 0), 1)

        for i in range(5, 15, 2):
            cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))

cv2.imwrite("img/out.jpg", draw)

cv2.imshow("test", draw)
cv2.waitKey(0)
