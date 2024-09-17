import numpy as np
import scipy as sp
import pylab


def get_diff(test_data, a):
    x = np.vstack(test_data[:, 0])
    y = np.vstack(test_data[:, 1])
    fit_y = np.dot(x, a)
    return np.sum((fit_y - y) ** 2, axis=1)


class least_squares:

    def __init__(self, data):
        self.data = data

    def fit(self):
        # N行2列的数据 第一列为X  第二列为Y
        x = np.vstack(self.data[:, 0])
        y = np.vstack(self.data[:, 1])
        # 调用最小二乘法函数 返回a
        x, residues, rank, s = sp.linalg.lstsq(x, y)
        return x

    """
    计算随机内群外数据的残差平方和
    """


class ransac:
    def __init__(self, data, n, k, t, d, return_all=False):
        self.data = data
        self.n = n
        self.k = k
        self.t = t
        self.d = d
        self.return_all = return_all

    def compute(self):
        num = 1
        best_diff = np.inf
        bestfit = None
        best_inner_idxs = None

        while num < self.k:
            idx1, idx2 = self.random_partition()
            # 计算随机内群数据的斜率
            ls = least_squares(self.data[idx1])
            x = ls.fit()
            # 将剩余数据代入获取残差平方和
            cal_data = get_diff(self.data[idx2], x)
            # 和阈值进行比较 获取满足条件的数据
            standard_data_idx = idx2[cal_data < self.t]
            num += 1
            if len(standard_data_idx) > self.d:
                # 合并数据 再进行最小二乘法计算
                concat_data = np.concatenate((self.data[idx1], self.data[standard_data_idx]))
                ls = least_squares(concat_data)
                x = ls.fit()
                concat_cal_data = get_diff(concat_data, x)
                # 计算平均残差平方和
                avg_diff = np.mean(concat_cal_data)
                if avg_diff < best_diff:
                    bestfit = x
                    best_diff = avg_diff
                    best_inner_idxs = np.concatenate((idx1, standard_data_idx))
        if bestfit is None:
            raise ValueError("did't meet fit acceptance criteria")
        if self.return_all:
            return bestfit, {'inliers': best_inner_idxs}
        else:
            return bestfit

    """
    数据切分为随机内群和剩余数据
    """
    def random_partition(self):
        idx = np.arange(self.data.shape[0])
        np.random.shuffle(idx)
        idx1 = idx[:self.n]
        idx2 = idx[self.n:]
        return idx1, idx2


if __name__ == "__main__":
    # 随机生成500个数据
    a_data = 20 * np.random.random((500, 1))
    # 乘以随机斜率
    perfect_fit = 60 * np.random.normal(size=(1, 1))
    b_data = np.dot(a_data, perfect_fit)

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = a_data + np.random.normal(size=a_data.shape)
    B_noisy = b_data + np.random.normal(size=b_data.shape)

    # 获取索引0-499
    all_idxs = np.arange(A_noisy.shape[0])
    # 打乱
    np.random.shuffle(all_idxs)
    # 100个0-500的随机局外点
    outlier_idxs = all_idxs[:100]
    A_noisy[outlier_idxs] = 20 * np.random.random((100, 1))  # 加入噪声和局外点的Xi
    B_noisy[outlier_idxs] = 60 * np.random.normal(size=(100, 1))  # 加入噪声和局外点的Yi
    all_data = np.hstack((A_noisy, B_noisy))
    ra = ransac(all_data, 50, 1000,  7e3, 300, True)
    ransac_fit, ransac_data = ra.compute()
    print("success", ransac_fit)


    linear_fit,resids,rank,s = sp.linalg.lstsq(np.vstack(all_data[:,0]), np.vstack(all_data[:,1]))
    if 1:


        sort_idxs = np.argsort(a_data[:, 0])
        A_col0_sorted = a_data[sort_idxs]  # 秩为2的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()

