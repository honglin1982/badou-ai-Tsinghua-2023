import numpy as np
import scipy as sp
import scipy.linalg as sl
import pylab

class LeastLinearSquareMethod(object):

    def __init__(self, input_columns, output_columns):
        self.input_columns = input_columns
        self.output_columns = output_columns

    def fit(self, data):
        # 获取xi
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        # 获取yi
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        fit = sl.lstsq(A, B)[0]
        return fit
    
    def get_error(self, fit, data):
        # xi
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        # yi
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # h(xi)
        B_fit = np.dot(A, fit)
        # 每点方差
        err_per_point = np.sum((B - B_fit)**2, axis=1)
        return err_per_point

def random_partition(n, n_data):
    """随机抽样

    Args:
        n (int): 抽样的个数
        n_data (ndarray的行数): 样本集的行数
    return:
        
    """
    # 获取全部索引
    all_idxs = np.arange(n_data)
    # 打乱索引
    np.random.shuffle(all_idxs)
    # 抽样的数据的索引
    random_idxs = all_idxs[:n]
    # 其余的索引为测试集的索引
    test_idxs = all_idxs[n:]
    return random_idxs, test_idxs

def ransac(data: np.ndarray, model: LeastLinearSquareMethod, n, t, k, d):
    """ransac算法

    Args:
        data (np.ndarray): 样本集
        model (LeastLinearSquareMethod): 最小二乘法模型
        n (int): 随机内群数量
        t (number): 每点方差最小阈值
        k (int): 迭代次数
        d (int): 最小内群点数量阈值

    Raises:
        ValueError: _description_

    Returns:
        turple: （ransac_fit, 最佳内群点的索引）
    """

    '''设置一些初始值'''
    iterations = 0
    besterr = np.inf
    best_inliers = None
    best_fit = None

    while iterations < k:
        # 随机抽样n个（xi，yi）为内群点
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        # 随机抽样点的值和测试集的值
        maybe_inliers = data[maybe_idxs, :]
        test_points = data[test_idxs, :]
        # 用抽样的内群点训练得到一个斜率
        maybefit = model.fit(maybe_inliers)
        # 用此斜率用在测试集上，得到每个点的方差
        testerr = model.get_error(maybefit, test_points)
        # 若某点的方差小于阈值t，那么认为这个点也是内群点
        also_inliers_idxs = test_idxs[testerr < t]
        also_inliers = data[also_inliers_idxs]
        # 若通过测试集得到的也是内群点的个数大于阈值d，进入下一步
        if len(also_inliers) > d:
            # 更新新的内群点集
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            # 训练
            better_fit = model.fit(betterdata)
            # 计算每点方差
            bettererr = model.get_error(better_fit, betterdata)
            # 计算均方差
            thiserr = np.mean(bettererr)
            if thiserr < besterr:
                # 满足条件后更新最好的标准
                besterr = thiserr
                best_fit = better_fit
                # 选用索引，方便以后画图
                best_inliers = np.concatenate((maybe_idxs, also_inliers_idxs))
        iterations += 1
    if best_inliers is None:
        raise ValueError("未找到合适模型！")
    return best_fit, best_inliers



if __name__ == "__main__":
    # np.random.seed(12345)
    '''设置样本点'''

    # 设置样本点的个数
    n_samples = 500
    n_inputs = 1
    n_outputs = 1

    # 生成随机样本点的xi
    A_exact = 20*np.random.random((n_samples, n_inputs))
    # 生成一个随机斜率（-60， 60）
    perfect_fit = 60*np.random.normal(size=(n_inputs, n_outputs))
    # 计算得出yi
    B_exact = np.dot(A_exact, perfect_fit)
    
    # 加噪声
    A_noisy = A_exact + np.random.normal(size=(n_samples, n_inputs))
    B_noisy = B_exact + np.random.normal(size=(n_samples, n_outputs))

    '''添加局外点'''

    # 设置局外点个数
    n_outliers = 100
    # 获取数据的全部索引(******使用np.arange, 不然后面会不匹配（testerr<t处）)
    dataidxs = np.arange(A_noisy.shape[0])
    # 打乱索引
    np.random.shuffle(dataidxs)
    # 局外点的索引
    outliers_idxs = dataidxs[:n_outliers]
    # 局外点重新赋值
    A_noisy[outliers_idxs] = 20*np.random.random((n_outliers, n_inputs))
    B_noisy[outliers_idxs] = 50*np.random.normal(size=(n_outliers, n_outputs))

    '''整理样本点'''
    # 合并（xi，yi）
    all_data = np.hstack((A_noisy, B_noisy))
    # 传入最小二乘法模型的参数，分别表示输入列xi，输出列yi
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]

    '''计算合适模型'''
    
    # 直接通过最小二乘法算出的斜率
    lsm_fit = sl.lstsq(A_noisy, B_noisy)[0]
    # 设置模型
    model = LeastLinearSquareMethod(input_columns, output_columns)
    # 通过ransac算出的斜率
    ransac_fit, ransac_inliers = ransac(all_data, model, 50, 7e3, 1000, 300)

    print('best_fit_ransac= ', ransac_fit)

    '''画图'''
    # 下面两行代码表示将xi从大到小排序，
    # 这是因为有可能不是所有的画图函数都默认将x轴从小到大排序
    # 本例无用处
    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]


    # 画出原始数据和ransac内群点数据的离散点图
    pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
    pylab.plot(A_noisy[ransac_inliers, :], B_noisy[ransac_inliers, :], 'cx', label='ransac data')
    
    # 绘制三条线
    pylab.plot(A_exact[:, 0], np.dot(A_exact, perfect_fit)[:, 0], 
               color='r', label='perfect_fit')
    pylab.plot(A_exact[:, 0], np.dot(A_exact, ransac_fit)[:, 0], 
               color='b', label='ransac_fit')
    pylab.plot(A_exact[:, 0], np.dot(A_exact, lsm_fit)[:, 0], 
               color='y', label='lsm_fit')
    pylab.legend()
    pylab.show()







