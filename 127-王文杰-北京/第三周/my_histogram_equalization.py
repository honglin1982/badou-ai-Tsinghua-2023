import cv2
import numpy

def get_hist(p_img_src, p_height, p_width):
    ret_hist = dict()
    for h in range(0,p_height):
        for w in range(0,p_width):
            p = p_img_src[h,w]
            if(p in ret_hist):
                ret_hist[p] += 1
            else:
                ret_hist[p] = 1
    return ret_hist

def get_resp_level(p_hist_src, p_height, p_width):
    sorted_hist = sorted(p_hist_src.items(), key=lambda x:x[0])
    sum_pi = 0
    q_dict = dict()
    for ite in sorted_hist:
        sum_pi += ite[1]/(p_height*p_width)
        q = sum_pi*256-1
        q_dict.update({ite[0]:int(q)})
        #print('key:%s,value:%s' % (ite[0],ite[1]))
    return q_dict

img_src = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
height,width = img_gray.shape
hist_gray = get_hist(img_gray, height, width)
img_dst = numpy.zeros((height,width),numpy.uint8)
resp_level = get_resp_level(hist_gray,height,width)
for i in range(0, height):
    for j in range(0, width):
        p = img_gray[i,j]
        if(p in resp_level):
            q = resp_level[p]
            img_dst[i,j] = q
cv2.imshow('img_gray--img_dst',numpy.hstack([img_gray,img_dst]))
cv2.waitKey(0)