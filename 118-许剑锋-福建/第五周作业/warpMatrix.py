'''
透视变换
'''

import numpy as np
import cv2


def warpMatrix(src, dst):
    nums = src.shape[0]
    a = np.zeros((2 * nums, 8))
    b = np.zeros((2 * nums, 1))
    for i in range(nums):
        a_i = src[i,:]
        b_i = dst[i,:]
        a[2*i, :]= [a_i[0], a_i[1], 1, 0, 0, 0, -a_i[0]*b_i[0], -a_i[1]*b_i[0]]
        b[2*i] = b_i[0]
        a[2 * i+1, :] = [0, 0, 0, a_i[0], a_i[1], 1, -a_i[0] * b_i[1], -a_i[1] * b_i[1]]
        b[2 * i+1] = b_i[1]

    a = np.mat(a)
    warp_matrix = a.T * b
    # print(warp_matrix)
    # # 后处理
    warp_matrix = np.array(warp_matrix).T[0]
    warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1.0, axis=0)
    warp_matrix = warp_matrix.reshape((3, 3))
    return warp_matrix


def my_warp_matrix(src, dst):
    if src.shape != (4, 2) and dst.shape != (4, 2):
        return
    a = np.zeros((8, 8))
    b = np.zeros((8, 1))
    for i in range(src.shape[0]):
        x, y = src[i]
        X, Y = dst[i]
        a[2*i] = [x, y, 1, 0, 0, 0, -x * X, -y * X]
        a[2*i+1] = [0, 0, 0, x, y, 1, -x * Y, -y * Y]
        b[2*i] = X
        b[2*i+1] = Y


    warp_matrix = np.dot(a.T, b)
    # print(warp_matrix)
    # a33为1
    warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1.0, axis=0)
    # warp_matrix.reshape((3, 3))
    return warp_matrix.reshape((3, 3))





if __name__ == '__main__':
    src = np.array([[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]])
    dst = np.array([[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]])

    warp_matrix = warpMatrix(src, dst)
    my_warp_matrix = my_warp_matrix(src, dst)
    print(warp_matrix)
    print(my_warp_matrix)
