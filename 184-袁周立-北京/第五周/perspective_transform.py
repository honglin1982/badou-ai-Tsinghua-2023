import numpy as np


'''
实现CANNY 实现透视变换
'''
def perspective_transform(src, dst):
    A = np.zeros((8, 8))
    B = np.zeros((8, 1))
    for i in range(4):
        A[2 * i] = np.array([src[i][0], src[i][1], 1, 0, 0, 0, -src[i][0] * dst[i][0], -src[i][1] * dst[i][0]])
        A[2 * i + 1] = np.array([0, 0, 0, src[i][0], src[i][1], 1, -src[i][0] * dst[i][1], -src[i][1] * dst[i][1]])
        B[2 * i] = dst[i][0]
        B[2 * i + 1] = dst[i][1]
    warp_matrix = np.dot(np.matrix(A).I, B)
    warp_matrix = np.insert(warp_matrix, 8, values=1)
    return warp_matrix.reshape(3, 3)


src = np.array([[1, 2, 3],
                [2, 3, 4],
                [111, 232, 557],
                [12, 357, 889]])
dst = np.array([[1, 2, 5],
                [3, 111, 7],
                [246, 769, 33],
                [222, 643, 234]])
print(perspective_transform(src, dst))