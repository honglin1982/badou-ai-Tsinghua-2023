import numpy as np

def WarpPerspectiveMatrix(src, dst):
    # 必须满足以下条件
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    
    num = src.shape[0]
    A = np.zeros((2*num, 2*num))
    B = np.zeros((2*num, 1))
    for i in range(num):
        A_i = src[i, :]
        B_i = dst[i, :]
        # 奇数排的A
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        # 偶数排的A
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        # 奇数排的B
        B[2*i] = B_i[0]
        # 偶数排的B
        B[2*i+1] = B_i[1]
    
    # A x warpMatrix = B, warpMatrix = A**(-1) * B
    A = np.mat(A)
    warpMatrix = A.I * B # 已经算出了a11到a32, 是一个列向量, 下面步骤整理成矩阵, 并且插入a33=1
    
    '''此时的warpMatrix为从上到下a11, a12, a13, a21, a22, a23, a31, a32的值,是一个列向量,还不是我们的转换矩阵'''
    warpMatrix = np.array(warpMatrix).T[0]  # 横起来
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0)  # 插入a33=1
    warpMatrix = warpMatrix.reshape(3, 3)
    return warpMatrix

if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)
    
    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)

    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
    