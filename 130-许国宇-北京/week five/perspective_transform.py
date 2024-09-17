import cv2
import numpy as np

class Transform:
    def __init__(self,image_path):
        self.image = cv2.imread(image_path)
        self.image_width = self.image.shape[:2][0]
        self.image_height = self.image.shape[:2][1]
        self.image_channels = self.image.shape[2]
    def PerspectiveTransform(self,src,dst):
        #获取透视变换矩阵
        self.PerspectiveMatrix(src,dst)
        a11=self.warpMatrix[0][0]
        a12=self.warpMatrix[0][1]
        a13=self.warpMatrix[0][2]
        a21=self.warpMatrix[1][0]
        a22=self.warpMatrix[1][1]
        a23=self.warpMatrix[1][2]
        a31=self.warpMatrix[2][0]
        a32=self.warpMatrix[2][1]
        max_i = 0
        min_i = 0
        max_j = 0
        min_j = 0
        for i in range(self.image_width):
            for j in range(self.image_height):
                result_i=int((a11*i+a12*j+a13)/(a31*i+a32*j+1))
                result_j=int((a21*i+a22*j+a23)/(a31*i+a32*j+1))
                if result_i<=min_i:
                    min_i=result_i
                if result_i>max_i:
                    max_i=result_i
                if result_j<=min_j:
                    min_j=result_j
                if result_j>max_j:
                    max_j=result_j
        self.image_result=np.zeros([(abs(min_i)+abs(max_i)+1),(abs(min_j)+abs(max_j)+1),3],dtype=np.uint8)
        for i in range(self.image_width):
            for j in range(self.image_height):
                result_i = int((a11 * i + a12 * j + a13) / (a31 * i + a32 * j + 1))+abs(min_i)
                result_j = int((a21 * i + a22 * j + a23) / (a31 * i + a32 * j + 1))+abs(min_j)
                self.image_result[result_i,result_j]=[self.image[i,j][k] for k in range(3)]
        cv2.imshow("origin image",self.image)
        cv2.imshow('result image',self.image_result)
        cv2.waitKey()
    def PerspectiveMatrix(self,src,dst):
        assert src.shape[0]==dst.shape[0] and src.shape[0]>=4
        num=src.shape[0]
        A=np.zeros((2*num,8))
        B=np.zeros((2*num,1))
        for i in range(0, num):
            A_i = src[i, :]
            B_i = dst[i, :]
            A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
            B[2 * i] = B_i[0]
            A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
            B[2 * i + 1] = B_i[1]
        A = np.mat(A)
        # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
        warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

        # 之后为结果的后处理
        warpMatrix = np.array(warpMatrix).T[0]
        warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
        warpMatrix = warpMatrix.reshape((3, 3))
        self.warpMatrix=warpMatrix

if __name__=="__main__":
    imagePath = "D:\BaDou\cv\Week Five\watch.jpg"
    transform = Transform(imagePath)
    dst = np.float32([[46, 169], [324, 43], [550, 493], [264, 624]])
    src = np.float32([[150, 50], [455, 50], [455, 500], [150, 500]])
    transform.PerspectiveTransform(src,dst)