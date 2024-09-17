import cv2
import matplotlib.pyplot as plt
import numpy as np

src = np.float32([[71, 192], [276, 180], [37, 350], [341, 326]])
dst = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
img = cv2.imread("resource.jpg")
n_img = cv2.resize(img, (0, 0), None, 0.3, 0.3)
n_img = cv2.cvtColor(n_img, cv2.COLOR_BGR2RGB)
plt.figure(1)
plt.imshow(n_img)


m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(n_img, m, (400, 400))
plt.figure(2)
plt.imshow(result)
plt.show()




