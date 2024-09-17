import numpy as np


def warpPerspactiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]

    A = np.zeros((2 * nums, 8))
    B = np.zeros((2 * nums, 1))

    for i in range(nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        B[2 * i] = B_i[0]
        B[2 * i + 1] = B_i[1]

        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]

    print(A)
    print(B)
    A = np.mat(A)
    warpMatrix = A.I * B
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    # print(warpMatrix, warpMatrix.shape)

    return warpMatrix.reshape((3, 3))

    # print(A, B)


if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    warpMatrix = warpPerspactiveMatrix(src, dst)
    # warpPerspactiveMatrix(src, dst)
    print(warpMatrix)

