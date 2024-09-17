import numpy as np
import cv2

def WarpMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] > 3

    rowLen = src.shape[0]
    A = np.zeros((2 * rowLen, 8))
    B = np.zeros((2 * rowLen, 1))
    for i in range(rowLen):
        Ai, Bi = src[i,:], dst[i,:]
        A[2 * i,:] = [Ai[0], Ai[1], 1, 0, 0, 0,
                      -Ai[0] * Bi[0], -Ai[1] * Bi[0]]
        B[2 * i] = Bi[0]
        A[2 * i + 1, :] = [0, 0, 0,Ai[0], Ai[1], 1,
                       -Ai[0] * Bi[1], -Ai[1] * Bi[1]]
        B[2 * i + 1] = Bi[1]
    A = np.mat(A)
    warpMatrix = A.I * B
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    warpMatrix = warpMatrix.reshape(3, 3)
    return warpMatrix

if __name__ == '__main__':

    img = cv2.imread('lenna.png')
    res = img.copy()
    src = np.float32([[511, 511], [511, 0], [0, 511], [0, 0]])
    dst = np.float32([[0, 0], [0, 511], [511, 0], [511, 511]])
    m1 = cv2.getPerspectiveTransform(src, dst)
    print('warpMatrix1:\n',  m1)
    m2 = WarpMatrix(src, dst)
    print('warpMatrix2:\n', m2)
    res = cv2.warpPerspective(res, m2, (512, 512))
    cv2.imshow('src', img)
    cv2.imshow('res', res)
    cv2.waitKey(0)