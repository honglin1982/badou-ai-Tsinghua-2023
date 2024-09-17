# 透视变换代码思路：利用原图和目标图已知的四个对应点，求出转换矩阵，并以原图为输入进行目标图的输出。
# 结合老师发的warpMatrix代码和透视变换代码，写在一起了。但是没有运行出来==

import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
result3 = img.copy()  #此处浅拷贝是为了之后使用result3时不改变img

def WarpPerspectiveMatrix(src, dst):
    # 断言语句，用于判断源点集和目标点集是否符合透视变换的要求.行数应该相等,应该至少包含四个点
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2*nums, 8))
    B = np.zeros((2*nums, 1))
    for i in range(nums):
        A_i = src[i, :]
        B_i = src[i, :]
        # 四个方程组
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]

    A = np.mat(A)  # 将数组转换为矩阵类型
    warpMatrix = A.I * B  # 求A的逆矩阵A.I，与B相乘。求出a11到a32

    # 矩阵后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  #插入a33=1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


if __name__ == '__main__':
    src = [[207, 151], [517, 285], [17, 601], [343, 731]]
    src = np.array(src)
    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    print('WarpMatrix')
    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
    result = cv2.warpPerspective(result3, warpMatrix, (337, 488))
    cv2.imshow("src", img)
    cv2.imshow("dst", result)
    cv2.waitKey(0)
