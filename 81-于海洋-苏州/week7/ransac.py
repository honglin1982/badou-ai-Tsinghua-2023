import numpy as np
import pylab
import scipy as sp
import scipy.linalg as sl


class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        _x, _, _, _ = sl.lstsq(A, B)  # residues:残差和
        return _x  # 返回最小平方和向量

    def get_error(self, data, _model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = np.dot(A, _model)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point


def ransac(data: np.ndarray, _m, min_count, max_iter, t, d, return_all=False):
    """

    :param data: 原始数据组
    :param _m: model
    :param min_count: 生成样本最少点
    :param max_iter: 最大迭代次数
    :param t: 阈值:作为判断点满足模型的条件
    :param d: 拟合较好时,需要的样本点最少的个数,当做阈值看待
    :param return_all: 返回所有数据
    """
    iterations = 0
    bestfit = None
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None
    while iterations < max_iter:
        # Step1. 在数据中随机选择几个点设定为内群
        # maybe_idx 进行计算的点
        # test_idx Step3 中进行验证的点
        maybe_idx, test_idx = random_partition(min_count, data.shape[0])
        maybe_inliers = data[maybe_idx, :]  # 获取size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idx]  # 若干行(Xi,Yi)数据点
        # Step2. 计算适合内群的点
        maybemodel = _m.fit(maybe_inliers)  # 拟合模型
        # Step3. 把其它刚才没选到的点带入刚才建立的模型中，计算是否为内群
        test_err = _m.get_error(test_points, maybemodel)  # 计算误差:平方和最小
        # Step4. 记下内群数量
        also_idx = test_idx[test_err < t]
        also_inliers = data[also_idx, :]
        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
            bettermodel = _m.fit(betterdata)
            better_errs = _m.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            # Step6. 保留最优解
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idx, also_idx))  # 更新局内点,将新点加入
        iterations += 1
    if bestfit is None:
        raise ValueError("未发现合适参数")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    all_idx = np.arange(n_data)  # 获取n_data下标索引
    np.random.shuffle(all_idx)  # 打乱下标索引
    idx_1 = all_idx[:n]  # 获取前n个数据
    idx_2 = all_idx[n:]  # 获取从第n个以后的数据
    return idx_1, idx_2


def gen_arr(count=3, replace_count=1, inp=1, out=1):
    """
    y = k * x + b
    """
    arr_k = 60 * np.random.normal(size=(inp, out))
    arr_x = 20 * np.random.random((count, inp))
    arr_y = np.dot(arr_x, arr_k)  # y = x * k + b

    x_noisy = arr_x + np.random.normal(size=arr_x.shape)
    y_noisy = arr_y + np.random.normal(size=arr_y.shape)

    arr_idx = np.arange(x_noisy.shape[0])  # 获取索引0-499
    np.random.shuffle(arr_idx)  # 打乱索引
    replace_index = arr_idx[:replace_count]  # 取替换点
    # 替换点
    x_noisy[replace_index] = 20 * np.random.random((replace_count, inp))
    y_noisy[replace_index] = 50 * np.random.normal(size=(replace_count, out))

    data = np.hstack((x_noisy, y_noisy))
    return data, arr_x, arr_k


def gen_model(inp, out, dug=False):
    input_c = range(inp)  # 数组的第一列x:0
    output_c = [inp + i for i in range(out)]  # 数组最后一列y:1
    m = LinearLeastSquareModel(input_c, output_c)
    return m


def display():
    sort_index = np.argsort(src_x[:, 0])
    src_sorted = src_x[sort_index]  # 秩为2的数组

    # 画点
    pylab.plot(x[:, 0], y[:, 0], 'k.', label='Origin data')  # 散点图
    pylab.plot(x[ransac_data['inliers'], 0], y[ransac_data['inliers'], 0], 'bx', label="Ransac remark")

    # 画线
    pylab.plot(src_sorted[:, 0], np.dot(src_sorted, src_k)[:, 0], label='Actual fit')
    pylab.plot(src_sorted[:, 0], np.dot(src_sorted, linear_fit)[:, 0], label='Lstsq fit')
    pylab.plot(src_sorted[:, 0], np.dot(src_sorted, ransac_fit)[:, 0], label='Ransac fit')
    pylab.legend()
    pylab.show()


if __name__ == "__main__":
    sample_count = 500
    rps_count = 100
    input_num = 1
    output_num = 1
    debug = False
    arr_data, src_x, src_k = gen_arr(count=sample_count, replace_count=rps_count, inp=input_num, out=output_num)
    model = gen_model(inp=input_num, out=output_num, dug=debug)

    x = arr_data[:, 0].reshape(-1, 1)
    y = arr_data[:, 1].reshape(-1, 1)

    # SciPy 库中最小二乘法
    # 返回值：
    # x：形状为 (N,) 或 (N, K) 的数组，表示线性方程组的最小二乘解。
    # residuals：形状为 () 或 (1,) 或 (K,) 的数组，表示残差平方和。
    # rank：矩阵 x 的秩。
    # s：形状为 (min(M, N),) 的奇异值数组。
    linear_fit, residuals, rank, s = sp.linalg.lstsq(x, y)
    ransac_fit, ransac_data = ransac(arr_data, model, 50, max_iter=10, t=7e3, d=300, return_all=True)

    display()
