import numpy as np
import cv2


def getwarpMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    num = src.shape[0]
    A = np.zeros((num*2, 8))
    B = np.zeros((num*2, 1))
    # 根据公式A * warpMatrix = B 构建矩阵A和B
    for i in range(num):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[i*2, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                     -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[i*2, :] = B_i[0]

        A[i*2+1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                       -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[i*2+1, :] = B_i[1]

    A = np.mat(A)   # 创建A矩阵
    warpMatrix = A.I * B    # 用A.I求出A的逆矩阵，然后与B相乘，求出前8个warpMatrix的未知数
    print('warpMatrix0:\n', warpMatrix, warpMatrix.shape)

    # 处理warpMatrix，生成3x3矩阵
    warpMatrix = np.array(warpMatrix).T[0]
    print('warpMatrix1:\n', warpMatrix, warpMatrix.shape)
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)     # 插入a_33=1
    print('warpMatrix2:\n', warpMatrix, warpMatrix.shape)
    warpMatrix = warpMatrix.reshape((3, 3))

    return warpMatrix


img = cv2.imread('IMG_3774.JPG')
cv2.imshow('Source', img)
cv2.waitKey()

src = np.float32([[896, 285], [568, 910], [517, 145], [110, 654]])
dst = np.float32([[0, 0], [720, 0], [0, 480], [720, 480]])

m0 = cv2.getPerspectiveTransform(src, dst)
m = getwarpMatrix(src, dst)
print('warpMatrix:\n', m0)

res = cv2.warpPerspective(img, m, (720, 480))
cv2.imshow('Result', res)
cv2.waitKey()
