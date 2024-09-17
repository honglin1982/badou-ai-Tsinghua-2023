import numpy as np


def warp_perspective_matrix(src: np.ndarray, dst: np.ndarray):
    """
    本来矩阵系数 是奇数，但是将最后的数值设置为1，就减少了需要求个的未知量
    """
    assert src.shape == dst.shape and src.shape[0] >= 4
    point_num = src.shape[0] * 2  # 表示坐标拆分出来求解的数目
    left = np.zeros((point_num, point_num))  # 定义矩阵x, y, xp, yp
    right = np.zeros((point_num, 1))  # xp, yp 结果作为一列矩阵
    for i in range(src.shape[0]):
        x = src[i, :][0]
        y = src[i, :][1]
        xp = dst[i, :][0]
        yp = dst[i, :][1]
        left[i * 2, :] = [x, y, 1, 0, 0, 0, -x*xp, -y*xp]
        right[i * 2, :] = xp
        left[i * 2 + 1, :] = [0, 0, 0, x, y, 1, -x*yp, -y*yp]
        right[i * 2 + 1, :] = yp
    left = np.mat(left)  # 方便转化成逆矩阵，inverse = numpy.linalg.inv(x), 或者是 转化成np.mat()，直接可以使用.I转成逆矩阵
    # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    warp_matrix = np.dot(left.I, right)  # right / left == 也就是 结果 乘以left的逆矩阵, 矩阵相乘最好用 np.dot，点相乘用np.multiply()
    # todo 逆矩阵也可使用如下方法
    # left = np.linalg.inv(left)
    # warp_matrix = np.dot(left, right)
    warp_matrix = np.array(warp_matrix).T[0]
    warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1.0, axis=0)
    return warp_matrix


if __name__ == '__main__':
    print('warpMatrix')
    # src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array([[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]])
    # dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array([[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]])
    warpMatrix = warp_perspective_matrix(src, dst)
    print(warpMatrix)
