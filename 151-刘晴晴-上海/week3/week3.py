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

from skimage.util.dtype import img_as_bool

class ImagePro:
    def CorrCVandPltDiff(self,image):
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
        #先归一化再放大，防止参数溢出，保证最小失真
        img_corr=image/255*(255/maxVal)*255
        return img_corr
    def CV2Show_in_one(self,images,colums,name):
         i_h, i_w,i_c = images[0].shape[:3]
         rows=len(images)/colums
         rows=math.ceil(rows)
         merge_img = np.zeros((i_h*rows,i_w*colums,i_c), images[0].dtype)
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
                  merge_img[t_h_start:t_h_end, t_w_start:t_w_end,0:i_c] = im
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
    def iamge_resize_1(self,image,scale,Type):     
         height,width,channels =image.shape       
         resize_height=int((height)*scale+0.5); 
         resize_width=int((width)*scale+0.5);
         if Type=="interface":
             ResizeImage=cv2.resize(image,(resize_height,resize_width))           
         elif Type=="near_1":
              #重新定义一下scale
              #由于数组是从0开始的，所以要先减1   
              #scale=(dst-1)/(src-1),
              #这样可以保证图像几何中心和4个顶点的位置匹配   
              #重新定义放大后的倍率          
             scale_x=(height-1)/(resize_height-1)
             scale_y=(width-1)/(resize_width-1)
             ResizeImage=np.zeros((resize_height,resize_width,channels),np.uint8)
             for z in range (3):
                for i in range(resize_height):
                    for j in range(resize_width):
                        #计算对应放大倍率后对应的原图像的坐标
                        x=i*scale_x
                        y=j*scale_y
                        #计算距离原图最近的坐标
                        #四舍五入求最近的点
                         #使用int取整，不管小数是多少，都会强制舍弃，
                         #所以将计算后的值+0.5，保证图像的像素数是整数
                         #向上取整ceil函数
                         #四舍五入也可以选择round函数
                        x=int(x+0.5)
                        y=int(y+0.5)
                        if(x>=height-1):
                             x=height-1
                        if(y>=width-1):
                             y=width-1
                        ResizeImage[i,j,z]=image[x,y,z]
         elif Type=="bilinear_1":
             #几何中心和四角重合
             ResizeImage=np.zeros((resize_height,resize_width,channels),np.uint8)
             #重新定义一下scale
             scale_x=(height-1)/(resize_height-1)
             scale_y=(width-1)/(resize_width-1)
             for z in range (3):
                for i in range(resize_height):
                    for j in range(resize_width):
                        x=j*scale_x
                        y=i*scale_y                                              
                         #np.floor:对输入的数组全部向下取整
                         #将双线性的计算公式展开，统一换成乘法
                        src_x0 = int(x)
                        src_x1 = min(src_x0 + 1 ,height - 1)
                        src_y0 = int(y)
                        src_y1 = min(src_y0 + 1, width - 1)
                        temp0 = (src_x1 - x) * image[src_y0,src_x0,z] + (x - src_x0) * image[src_y0,src_x1,z]
                        temp1 = (src_x1 -x) * image[src_y1,src_x0,z] + (x - src_x0) * image[src_y1,src_x1,z]
                        ResizeImage[i,j,z] = int((src_y1 - y) * temp0 + (y - src_y0) * temp1)
         elif Type=="bilinear_2":
             #几何中心重合
             ResizeImage=np.zeros((resize_height,resize_width,channels),np.uint8)
             scale=1/scale
             for z in range (3):
                for i in range(resize_height):
                    for j in range(resize_width):                      
                        y = (i + 0.5) * scale-0.5
                        x = (j + 0.5) * scale-0.5                    
                # find the coordinates of the points which will be used to compute the interpolation
                        src_x0 = int(x)
                        src_x1 = min(src_x0 + 1 ,width - 1)
                        src_y0 = int(y)
                        src_y1 = min(src_y0 + 1, height - 1)
                        temp0 = (src_x1 - x) * image[src_y0,src_x0,z] + (x - src_x0) * image[src_y0,src_x1,z]
                        temp1 = (src_x1 -x) * image[src_y1,src_x0,z] + (x - src_x0) * image[src_y1,src_x1,z]
                        ResizeImage[i,j,z] = int((src_y1 - y) * temp0 + (y - src_y0) * temp1)
                        print(scale,x,y,i,j,temp0,temp1,ResizeImage[i,j,z])
         #print("image show gray: %s"% ResizeImage)
         print(image.shape[:2])
         print(ResizeImage.shape[:2])
         return ResizeImage
    def iamge_resize_2(self,image,scale1,scale2,Type):      
         height,width,channels =image.shape
         #由于数组是从0开始的，所以要先减1   
         #scale=(dst-1)/(src-1),
         #这样可以保证图像几何中心和4个顶点的位置匹配   
         #放大后的尺寸
         resize_height=int((height)*scale1+0.5); 
         resize_width=int((width)*scale2+0.5);
         if Type=="interface":
             ResizeImage=cv2.resize(image,(resize_height,resize_width))           
         elif Type=="near_1":
             #重新定义一下scale
             scale_x=(height-1)/(resize_height-1)
             scale_y=(width-1)/(resize_width-1)
             ResizeImage=np.zeros((resize_height,resize_width,channels),np.uint8)
             for z in range (3):
                for i in range(resize_height):
                    for j in range(resize_width):
                        x=i*scale_x
                        y=j*scale_y
                        #四舍五入求最近的点
                         #使用int取整，不管小数是多少，都会强制舍弃，
                         #所以将计算后的值+0.5，保证图像的像素数是整数
                         #向上取整ceil函数
                         #四舍五入也可以选择round函数
                        x=int(x+0.5)
                        y=int(y+0.5)
                        if(x>=height-1):
                             x=height-1
                        if(y>=width-1):
                             y=width-1
                        ResizeImage[i,j,z]=image[x,y,z]
         elif Type=="bilinear_1":
             #重新定义一下scale
             scale_x=(height-1)/(resize_height-1)
             scale_y=(width-1)/(resize_width-1)
             ResizeImage=np.zeros((resize_height,resize_width,channels),np.uint8)
             for z in range (3):
                for i in range(resize_height):
                    for j in range(resize_width):
                        x=i*scale_x
                        y=j*scale_y                                              
                         #np.floor:对输入的数组全部向下取整
                         #将双线性的计算公式展开，统一换成乘法
                        src_x0 = int(x)
                        src_x1 = min(src_x0 + 1 ,height - 1)
                        src_y0 = int(y)
                        src_y1 = min(src_y0 + 1, width - 1)
                        temp0 = (src_x1 - x) * image[src_y0,src_x0,z] + (x - src_x0) * image[src_y0,src_x1,z]
                        temp1 = (src_x1 -x) * image[src_y1,src_x0,z] + (x - src_x0) * image[src_y1,src_x1,z]
                        ResizeImage[i,j,z] = int((src_y1 - y) * temp0 + (y - src_y0) * temp1)
         elif Type=="bilinear_2":
             ResizeImage=np.zeros((resize_height,resize_width,channels),np.uint8)
             for z in range (3):
                for i in range(resize_height):
                    for j in range(resize_width):
                        scale=1/scale
                        x = (i + 0.5) * scale-0.5
                        y = (j + 0.5) * scale-0.5
                # find the coordinates of the points which will be used to compute the interpolation
                        src_x0 = int(x)
                        src_x1 = min(src_x0 + 1 ,height - 1)
                        src_y0 = int(y)
                        src_y1 = min(src_y0 + 1, width - 1)
                        temp0 = (src_x1 - x) * image[src_y0,src_x0,z] + (x - src_x0) * image[src_y0,src_x1,z]
                        temp1 = (src_x1 -x) * image[src_y1,src_x0,z] + (x - src_x0) * image[src_y1,src_x1,z]
                        ResizeImage[i,j,z] = int((src_y1 - y) * temp0 + (y - src_y0) * temp1)

         #print("image show gray: %s"% ResizeImage)
         print(image.shape[:2])
         print(ResizeImage.shape[:2])
         return ResizeImage
    def images_scale(self,images,scale):
        images_List=[]
        h,w=images[0].shape[:2]
        for img in images:
            scale_img=cv2.resize(img,(int(h*scale),int(w*scale)))
            images_List.append(scale_img)
        return images_List 
    def his_equalization(self,image):
       c=len(image.shape)
       imp=ImagePro()
       if(c==3):
          #彩色图像先对三个通道分别均衡化，再合并,会导致失真严重，因为均衡化并不是线性的变化
          #会导致RGB三个分量的变化不再线性，即RGB的分量比例会发生改变，重新合并就会导致色相发生变化
          #b,g,r=cv2.split(image)
          #b=imp.equalization_1(b)
          #g=imp.equalization_1(g)
          #r=imp.equalization_1(r)
          #image=cv2.merge([r,g,b])
          #故彩色图像均衡化，采用先将图片转换成YCbCr模式或者LAB等可以把亮度分离出来的模式，
          #针对Y通道进行均值化，再合并，可以避免色相发生改变，而仅是亮度发生变化
          #以转换成YUV模式为例
          image_YUV=cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
          #回头补各个色彩模型转换的代码实现
          Y,U,V=cv2.split(image_YUV)
          L=imp.equalization_1(Y)
          image=cv2.merge([L,U,V])
          #最后还要转换成BGR的格式，显示才不出错
          image=cv2.cvtColor(image,cv2.COLOR_YUV2BGR)
       else:
           #image=cv2.equalizeHist(image)    
           image=imp.equalization_1(image)     
       return image
    def equalization_1(self,image):
         #灰度图片的直方图均衡化
         c=len(image.shape)
         if(c==3):
           print("单一通道均值化不适合彩色图片")
           return image;
         else:
           h,w=image.shape[:2]
           array=np.zeros([256], dtype = np.int32)
           #获取每一个亮度级的像素数目
           for row in range(h):
               for col in range(w):
                   pix_value=image[row,col]
                   array[pix_value]+=1
           #计算每一个亮度级的像素概率
           #print("array:\n %s"%array)
           prob=np.zeros([256], dtype = np.float32)
           for p in range(256):
               prob[p]=array[p]/(h*w)
           #累加每一个亮度级的像素概率
           #print("array:\n %s"% prob)
           sum_b=np.zeros([256], dtype = np.float32)
           for q in range(256):
               if(q!=0):
                   sum_b[q]=sum_b[q-1]+prob[q]
               else:
                   sum_b[q]=prob[q]
           #计算直方图均衡化的结果，即当前的亮度值*当前亮度值累加的像素概率
           for row in range(h):
               for col in range(w):
                   pix_value=image[row,col]
                   #image[row,col]=pix_value*sum_b[pix_value]
                   #得到的直方图很有特点，回头可以再研究一下，看有没有应用场景
                   #image[row,col]=256*sum_b[pix_value]-1
                   #归一化到255和CV的显示一致
                   #可以理解为本身图像用数字表示就是[0,255]共256阶，映射的最大值也是255，故乘以255即可
                   #乘上的值可以理解为我们想要映射的最大级数
                   image[row,col]=255*sum_b[pix_value]
           return image
     
class ImageShow:
    def image_resize_show(self):
        path="lena.jpg"
        img = cv2.imread(path)
        imp=ImagePro()
        image_resize1=imp.iamge_resize_1(img,0.2,'interface')
        image_resize2=imp.iamge_resize_1(img,0.2,'near_1')
        #image_resize3=imp.iamge_resize_1(img,5,'bilinear_1')
        #image_resize6=imp.iamge_resize_1(img,5,'bilinear_2')
        images_resize=[image_resize1,image_resize2]
        imp.CV2Show_in_one(images_resize,2,'images_rezie')
        cv2.imshow('image_resize1',image_resize1)
        #cv2.imshow('image_resize2',image_resize2)
        #cv2.imshow('image_resize3',image_resize3)
        #cv2.imwrite("1.jpg",img_gray)
        #cv2.imwrite("image_resize1.jpg",image_resize1)
        #cv2.imwrite("image_resize2.jpg",image_resize2)
        #cv2.imwrite("image_resize5.jpg",image_resize3)
        #cv2.imwrite("image_resize6.jpg",image_resize6)
    def image_equalization_show(self):
        path="lena.jpg"
        img = cv2.imread(path)
        imp=ImagePro()
        img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('ori',img_gray)
        cv2.imwrite('img_gray.jpg',img_gray)
        cv2.imshow('ori',img)
        hist_1=cv2.calcHist([img_gray],[0],None,[256],[0,255])
        image_equalization=imp.his_equalization(img)
        #image_equalization_gray=cv2.cvtColor(image_equalization, cv2.COLOR_BGR2GRAY)         
        cv2.imshow('his',image_equalization)       
        cv2.imwrite('image_equalization.jpg',image_equalization)
        hist_2=cv2.calcHist([image_equalization],[0],None,[256],[0,255])
        plt.plot(hist_1,color='b')
        plt.plot(hist_2,color='g')
        plt.show()

if __name__ == '__main__':
    Ims=ImageShow()
    Ims.image_resize_show()
    Ims.image_equalization_show()
    cv2.waitKey(0)

