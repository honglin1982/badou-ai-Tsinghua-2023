import cv2
import matplotlib.pyplot as plt
import numpy as np

# print(int(3.2))  3
# print(int(3.7))  3
'''
最邻近插值
img:原图
r_scale:缩放比例
'''
def near_insert(img,r_scale):
    h,w,c = img.shape
    r_height = int(h*r_scale)
    r_weight = int(w*r_scale)
    result_img = np.zeros((r_height,r_weight,c),dtype=img.dtype)
    for l in range(c):
        for i in range(r_height):
            for j in range(r_weight):
                x = int(i*h/r_height + 0.5)
                y = int(j*w/r_weight + 0.5)
                if x >= h:
                    x = h-1
                if y >= w:
                    y = w-1
                result_img[i,j,l] = img[x,y,l]
    return result_img

'''
双线性插值
img:原图
dstHeight,dstWeight:目标图像的h,w
'''
def bilinear_insert(img,dstHeight,dstWidth):
    # f(i + u, j + v) = (1 - u) (1 - v) f(i, j) + (1 - u) v f(i, j + 1) + u (1 - v) f(i + 1, j) + u v f(i + 1, j + 1)
    srcHeight,srcWidth,chanels = img.shape
    result_img = np.zeros((dstHeight,dstWidth,chanels),dtype=img.dtype)
    for i in range(dstHeight):
        for j in range(dstWidth):
            srcX = (i+0.5)*srcHeight/dstWidth - 0.5
            srcY = (j+0.5)*srcWidth/dstWidth - 0.5

            srcX_0 = int(np.floor(srcX))
            srcX_1 = int(np.ceil(srcX))
            u = float(srcX-srcX_0)
            if srcX_1 >= srcHeight:
                srcX_1 = srcX_1 - 1

            srcY_0 = int(np.floor(srcY))
            srcY_1 = int(np.ceil(srcY))
            v = float(srcY - srcY_0)
            if srcY_1 >= srcWidth:
                srcY_1 = srcY_1 - 1

            result_img[i,j]=(1-u)*(1-v)*img[srcX_0,srcY_0]+(1-u)*v*img[srcX_0,srcY_1]+u*(1-v)*img[srcX_1,srcY_0]+u*v*img[srcX_1,srcY_1]
    return result_img


img = cv2.imread("flower.png")
img2=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
w,h,c = img2.shape
near_img = near_insert(img2,2)
biliner_img = bilinear_insert(img2,1200,1400)
#放大,双线性插值
newimg0=cv2.resize(img2,(1400,1200),interpolation=cv2.INTER_LINEAR)
#放大,双立方插值
newimg1=cv2.resize(img2,(w*2,h*2),interpolation=cv2.INTER_CUBIC)
#放大, 最近邻插值
newimg2=cv2.resize(img2,(w*2,h*2),interpolation=cv2.INTER_NEAREST)
#放大, 象素关系重采样
newimg3=cv2.resize(img2,(w*2,h*2),interpolation=cv2.INTER_AREA)
#缩小, 象素关系重采样
newimg4=cv2.resize(img2,(300,200),interpolation=cv2.INTER_AREA)

plt.show()
plt.figure()
plt.subplot(241)
plt.imshow(img2)
plt.subplot(242)
plt.imshow(near_img)
plt.subplot(243)
plt.imshow(biliner_img)
plt.subplot(244)
plt.imshow(newimg0)
plt.subplot(245)
plt.imshow(newimg1)
plt.subplot(246)
plt.imshow(newimg2)
plt.subplot(247)
plt.imshow(newimg3)
plt.subplot(248)
plt.imshow(newimg4)
plt.show()
