#-*-coding:GBK -*-
from typing import Match
from matplotlib import image
from numpy.lib.shape_base import column_stack
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import math

class ImagePro:
    def RGB2GRAY(self,img,Type):
        if Type=="float":
            h,w = img.shape[:2]                               #获取图片的high和wide
            img_gray = np.zeros([h,w],img.dtype)   
            print("利用float转换gray:gray=B*0.11+G*0.59+R*0.3") 
            for i in range(h):
                 for j in range(w):
                     m = img[i,j]                             #取出当前high和wide中的BGR坐标
                     img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)   #将BGR坐标转化为gray坐标并赋值给新图像
            print("image show gray: %s"%img_gray)
            return img_gray
        elif Type=="int":
             h,w = img.shape[:2]                               #获取图片的high和wide
             img_gray = np.zeros([h,w],img.dtype)   
             print("利用int转换gray:gray=B*11+G*59+R*30") 
             for i in range(h):
                 for j in range(w):
                     m = img[i,j]                             #取出当前high和wide中的BGR坐标
                     img_gray[i,j] =int((m[0]*11 + m[1]*59 + m[2]*30)/100)    #将BGR坐标转化为gray坐标并赋值给新图像
             print("image show gray: %s"%img_gray)
             return img_gray
        elif Type=="Shift":
             h,w = img.shape[:2]                               #获取图片的high和wide
             img_gray = np.zeros([h,w],img.dtype)   
             print("利用移位转换gray:gray=（B*28+G*151+R*76）>>8") 
             for i in range(h):
                 for j in range(w):
                     m = img[i,j]                             #取出当前high和wide中的BGR坐标
                     img_gray[i,j] =int((m[0]*28 + m[1]*151 + m[2]*76)>>8)     #将BGR坐标转化为gray坐标并赋值给新图像
             print("image show gray: %s"%img_gray)
             return img_gray
        elif Type=="AVG_Nor":
             h,w = img.shape[:2]                               #获取图片的high和wide
             img_gray = np.zeros([h,w],img.dtype)   
             print("利用归一化后的值转换gray:gray=（B/255+G/255+R/255）/3") 
             for i in range(h):
                 for j in range(w):
                     m = img[i,j]                             #取出当前high和wide中的BGR坐标
                     img_gray[i,j] =int((m[0]/255 + m[1]/255 + m[2]/255)/3*255)     #将BGR坐标转化为gray坐标并赋值给新图像
             print("image show gray: %s"%img_gray)
             return img_gray
        elif Type=="AVG_UnNor":
             h,w = img.shape[:2]                               #获取图片的high和wide
             img_gray = np.zeros([h,w],img.dtype)   
             print("利用非归一化后的值转换gray:gray=（B+G+R）/3") 
             for i in range(h):
                 for j in range(w):
                     m = img[i,j]                             #取出当前high和wide中的BGR坐标
                     img_gray[i,j] =int((m[0] + m[1] + m[2])/3)     #将BGR坐标转化为gray坐标并赋值给新图像
             print("image show gray: %s"%img_gray)
             return img_gray
        elif Type=="G_Gray":
             h,w = img.shape[:2]                               #获取图片的high和wide
             img_gray = np.zeros([h,w],img.dtype)   
             print("利用绿色通道的值转换gray:gray=G") 
             for i in range(h):
                 for j in range(w):
                     m = img[i,j]                             #取出当前high和wide中的BGR坐标
                     img_gray[i,j] =int((m[1])/3)     #将BGR坐标转化为gray坐标并赋值给新图像
             print("image show gray: %s"%img_gray)
             return img_gray          
        elif   Type=="interface":
             h,w = img.shape[:2]                               #获取图片的high和wide
             img_gray = np.zeros([h,w],img.dtype)   
             print("利用rgb2gray转换gray:gray=rgb2grat(img)") 
             img_gray= rgb2gray(img)
             print("image show gray: %s"%img_gray)
             return img_gray*255
    def BinaryZation(img,threshold,value1,value2):
        h,w,c=img.shape[:3]
        img_gray = np.zeros([h,w],img.dtype)  
        if c==3:
           img_gray=cv2.cvtColor(img,cv2.COLOR_BayerGR2GRAY)
        else:
           img_gray=img
        if value1==0 and value2==0:
            print("使用接口进行二值化")
            r,img_BinZa=cv2.threshold(img_gray,threshold,255,cv2.THRESH_BINARY)
            #将灰度图转化为RGB
            img_array=img_BinZa.convert("RGB")
            return img_array
        else:
            print("使用定义值进行二值化")
            img_binaryzation=np.zeros([h,w],3)
            for i in range(h):
                 for j in range(w):
                      if (img_gray[i, j] <=threshold):
                        img_binaryzation[i, j] = value1
                      else:
                        img_binaryzation[i, j] = value2
            return img_binaryzation
    def CV2Show_in_one(self,images,colums,name):
         i_h, i_w = images[0].shape[:2]
         rows=len(images)/colums
         rows=math.ceil(rows)
         merge_img = np.zeros((i_h*rows,i_w*colums), images[0].dtype)

         max_count = len(images)
         count = 0
         for i in range(rows):
            if count >= max_count:
              break
            for j in range(colums):
               if count < max_count:
                  im = images[count]
                  t_h_start = i * (i_h)
                  t_w_start = j * (i_w)
                  t_h_end = t_h_start + im.shape[0]
                  t_w_end = t_w_start + im.shape[1]
                  merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                  count = count + 1
               else:
                 break
         cv2.imshow(name, merge_img)
    def pltShow_in_one(self,images,colums): 
 
         rows=(len(images)/colums)
         rows=math.ceil(rows)
         max_count = len(images)
         count = 0
         for i in range(rows):
            if count >= max_count:
              break
            for j in range(colums):
               if count < max_count:
                  im = images[count]
                  plt.subplot(rows,colums,count+1)
                  plt.imshow(im, cmap='gray')
                  plt.axis('off')  
                  count = count + 1
               else:
                 break
         plt.show()

    def images_scale(self,images,scale):
        images_List=[]
        h,w=images[0].shape[:2]
        for img in images:
            scale_img=cv2.resize(img,(int(h*scale),int(w*scale)))
            images_List.append(scale_img)
        return images_List 
    def CorrCVandPltDiff(self,image):
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
        #先归一化再放大，防止参数溢出，保证最小失真
        img_corr=image/255*(255/maxVal)*255
        return img_corr
if __name__ == '__main__':
    path="pic\\lena.jpg"
    img = cv2.imread(path)
    imp=ImagePro()
    image_gray1=imp.RGB2GRAY(img,'float')
    image_gray2=imp.RGB2GRAY(img,'int')
    image_gray3=imp.RGB2GRAY(img,'Shift')
    image_gray4=imp.RGB2GRAY(img,'AVG_Nor')
    image_gray5=imp.RGB2GRAY(img,'AVG_UnNor')
    image_gray6=imp.RGB2GRAY(img,'G_Gray')
    image_gray7=imp.RGB2GRAY(img,'interface')
	#关于图片5和6 ，CV2的imshow和plt显示不一致，
    #做下矫正
    image_gray5_corr=imp.CorrCVandPltDiff(image_gray5)
    image_gray6_corr=imp.CorrCVandPltDiff(image_gray6)
    images_corr=[image_gray1,image_gray2,image_gray3,image_gray4,image_gray5_corr,image_gray6_corr,image_gray7]
    images_scale_corr=imp.images_scale(images_corr,0.5)
    imp.CV2Show_in_one(images_scale_corr,3,'images_scale_corr')
    images=[image_gray1,image_gray2,image_gray3,image_gray4,image_gray5,image_gray6,image_gray7]
    images_scale=imp.images_scale(images,0.5)
    imp.CV2Show_in_one(images_scale,3,'images_scale')
    imp.pltShow_in_one(images,3)
    cv2.waitKey(0)
    cv2.destroyWindow()
   














    


   




    


