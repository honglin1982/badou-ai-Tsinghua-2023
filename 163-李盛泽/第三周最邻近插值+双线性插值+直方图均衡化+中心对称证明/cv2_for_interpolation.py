#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer


img = cv2.imread('lenna.png', 1)

start = timer()
dst_img1 = cv2.resize(img,(800,800),interpolation = cv2.INTER_NEAREST)
dst_img1_RGB = cv2.cvtColor(dst_img1,cv2.COLOR_BGR2RGB)
plt.subplot(121)
plt.title("1.最邻近插值法"),plt.axis('off')
plt.imshow(dst_img1_RGB)
print("with nearest_interp: ", timer() - start)

start = timer()
dst_img2 = cv2.resize(img,(800,800),interpolation = cv2.INTER_LINEAR)
dst_img2_RGB = cv2.cvtColor(dst_img2,cv2.COLOR_BGR2RGB)
plt.subplot(122)
plt.title("2.双线性插值法"),plt.axis('off')
plt.imshow(dst_img2_RGB)
print("with billinear_interp: ", timer() - start)

plt.savefig('cv2_for_interp_result.png')
plt.show()


'''
start = timer()
dst_img1 = cv2.resize(img,(800,800),interpolation = cv2.INTER_NEAREST)
print("with nearest_interp: ", timer() - start)

start = timer()
dst_img2 = cv2.resize(img,(800,800),interpolation = cv2.INTER_LINEAR)
print("with billinear_interp: ", timer() - start)

dst_img = np.hstack([dst_img1,dst_img2])
cv2.imshow("nearest/billinear",dst_img)
cv2.imwrite('cv2_for_interp_restult.jpg',dst_img,[int(cv2.IMWRITE_JPEG_QUALITY),70])

cv2.waitKey(0)
'''
