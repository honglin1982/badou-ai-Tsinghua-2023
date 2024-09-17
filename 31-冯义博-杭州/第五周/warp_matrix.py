import numpy as np


def parse(src, dst):
    if src.shape[0] != dst.shape[0] or src.shape[0] < 4:
        raise Exception("矩阵数据异常")
    num = src.shape[0]
    a = np.zeros((num * 2, num * 2))
    b = np.zeros((num * 2, 1))
    for i in range(num):
        a_i = src[i]
        b_i = dst[i]
        a[2*i, :] = [a_i[0], a_i[1], 1, 0, 0, 0, -a_i[0]*b_i[0], -a_i[1]*b_i[0]]
        a[2*i + 1, :] = [0, 0, 0, a_i[0], a_i[1], 1, -a_i[0]*b_i[1], -a_i[1]*b_i[1]]
        b[2*i] = b_i[0]
        b[2*i + 1] = b_i[1]

    a = np.mat(a)
    # a.I a的逆矩阵
    warpMatrix = a.I * b
    # 之后为结果的后处理
    # 矩阵转置取第一行
    warpMatrix = np.array(warpMatrix).T[0]
    # 末位补充1
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    # 转为3*3矩阵
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


if __name__ == "__main__":
    src = np.array([[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]])
    dst = np.array([[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]])
    matrix = parse(src, dst)
    print(matrix)
