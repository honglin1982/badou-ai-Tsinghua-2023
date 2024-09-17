import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

if __name__ == '__main__':

    # 1、灰度化
    pic_path = r'lenna.png'  #图片路径,r表示打开一个文件并返回一个文件对象，python自动管理关闭文件
    img = plt.imread(pic_path) #读取图片
    if pic_path[-4:] == '.png': #判断是否为PNG图片
        img = img * 255        #.png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
    img = img.mean(axis = -1)   ## 取均值就是灰度化了
    plt.figure('1、原图')
    plt.imshow(img.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')


    #2、高斯滤波
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    dim = int(np.round(6*sigma + 1)) # round是四舍五入函数，根据标准差求高斯核是几乘几的，也就是维度
    if dim%2 == 0:                   #卷积核必须是奇数
        dim = dim + 1
    Guassisan_filter = np.zeros([dim,dim]) # 存储高斯核，这是数组不是列表了
    tmp = [i-dim//2 for i in range(dim)]
    print(tmp)
    n1 = 1/(2*math.pi*sigma**2)  #高斯公式的分母,G(x,y)=12πσ2exp(−x2+y22σ2) 两个一维高斯的乘积,不是一维的G(x)
    n2 = -1/(2*sigma**2)         #高斯公式的指数的分母 
    for i in range(dim):
        for j in range(dim):
            Guassisan_filter[i,j] = n1 * math.exp(n2*(tmp[i]**2+tmp[j]**2))
    Guassisan_filter = Guassisan_filter/Guassisan_filter.sum()
    dx,dy = img.shape
    img_guass = np.zeros(img.shape)
    tmp = dim // 2
    img_pad = np.pad(img,((tmp,tmp),(tmp,tmp)),'constant') #边缘填补一圈
    for i in range(dx):
        for j in range(dy):
            img_guass[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*Guassisan_filter) #窗口滑动
    plt.figure('2、高斯滤波')
    plt.imshow(img_guass.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')
 

    #3、求梯度。以下两个是滤波求梯度用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1,0,1],
                               [-2,0,2],
                               [-1,0,1]])  #垂直方向sobel算子
    sobel_kernel_y = np.array([[1,2,1],
                               [0,0,0],
                               [-1,-2,-1]])#水平方向sobel算子
    img_tidu_x = np.zeros(img_guass.shape) #创建储存的图像
    img_tidu_y = np.zeros(img_guass.shape)
    img_tidu   = np.zeros(img_guass.shape)
    img_pad    = np.pad(img_guass,((1,1),(1,1)),'constant') #边缘添加，根据上面矩阵结构所以写1
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i,j] = np.sum(img_pad[i:i+3,j:j+3] * sobel_kernel_x)  #x方向卷
            img_tidu_y[i,j] = np.sum(img_pad[i:i+3,j:j+3] * sobel_kernel_y)  #y方向卷
            img_tidu[i,j]   = np.sqrt(img_tidu_x[i,j]**2 + img_tidu_y[i,j]**2) 

        img_tidu_x[img_tidu_x == 0] = 0.00000001  #使图像为0的地方置为一个很小的值，以便下一步除法
        angle =  img_tidu_y / img_tidu_x          #计算θ，给最邻近插值使用
    plt.figure('3、梯度')
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')
  
    
    #4、非极大值抑制
    img_NMS = np.zeros(img_tidu.shape)
    for i in range(1,dx-1):
        for j in range(1,dy-1):
            temp = img_tidu[i-1:i+2,j-1:j+2]
            if angle[i, j] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if  (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                img_NMS[i, j] = img_tidu[i, j]
    plt.figure('4、非极大值抑制')
    plt.imshow(img_NMS.astype(np.uint8), cmap='gray')
    plt.axis('off')

     # 5、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3
    zhan = []
    for i in range(1,img_NMS.shape[0]-1):
        for j in range(1,img_NMS.shape[1]-1):
            if img_NMS[i,j] >= high_boundary:     #大于高阈值的置255
                img_NMS[i,j] = 255
                zhan.append([i,j])                #记录强边缘
            elif img_NMS[i,j] <= lower_boundary:  #小于低阈值的置0
                img_NMS[i,j] = 0
    while not (len(zhan) == 0):
        temp1,temp2 = zhan.pop()                  #一个强边缘出栈
        a = img_NMS[temp1-1:temp1+2,temp2-1:temp2+2]  #强边缘的8邻域
        if (a[0,0] < high_boundary) and (a[0,0] >lower_boundary):
            img_NMS[temp1-1,temp2-1] = 255             
            zhan.append([temp1-1,temp2-1])             #与强边缘连接，置为强边缘 记录
        if (a[0,1] < high_boundary) and (a[0,1] >lower_boundary):
            img_NMS[temp1-1,temp2] = 255             
            zhan.append([temp1-1,temp2])             #与强边缘连接，置为强边缘 记录
        if (a[0,2] < high_boundary) and (a[0,2] >lower_boundary):
            img_NMS[temp1-1,temp2+1] = 255             
            zhan.append([temp1-1,temp2+1])             #与强边缘连接，置为强边缘 记录
        if (a[1,0] < high_boundary) and (a[1,0] >lower_boundary):
            img_NMS[temp1,temp2-1] = 255             
            zhan.append([temp1,temp2-1])             #与强边缘连接，置为强边缘 记录
        if (a[1,2] < high_boundary) and (a[1,2] >lower_boundary):
            img_NMS[temp1,temp2+1] = 255             
            zhan.append([temp1,temp2+1])             #与强边缘连接，置为强边缘 记录
        if (a[2,0] < high_boundary) and (a[2,0] >lower_boundary):
            img_NMS[temp1+1,temp2-1] = 255             
            zhan.append([temp1+1,temp2-1])             #与强边缘连接，置为强边缘 记录
        if (a[2,1] < high_boundary) and (a[2,1] >lower_boundary):
            img_NMS[temp1+1,temp2] = 255             
            zhan.append([temp1+1,temp2])             #与强边缘连接，置为强边缘 记录
        if (a[2,2] < high_boundary) and (a[2,2] >lower_boundary):
            img_NMS[temp1+1,temp2+1] = 255             
            zhan.append([temp1+1,temp2+1])             #与强边缘连接，置为强边缘 记录
   
    for i in range(img_NMS.shape[0]):              #剩下的点全部置0
        for j in range(img_NMS.shape[1]):
            if img_NMS[i,j] != 255 and img_NMS[i,j] != 0:
                img_NMS[i,j] = 0

    plt.figure('5、双阈值检测')
    plt.imshow(img_NMS.astype(np.uint8),cmap='gray')
    plt.axis('off')
    plt.show()









    

    
    






    

  
   


    






    
    
 
    
