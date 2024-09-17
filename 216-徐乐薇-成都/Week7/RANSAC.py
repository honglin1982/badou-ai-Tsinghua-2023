# RANSAC
# 1. 随机选择一个最小数据集设定为内群
# 2. 用内群拟合模型
# 3. 用模型测试所有其他数据，计算是否为内群
# 4. 记下内群数量
# 5. 重复上述步骤，直到找到最优模型
# 6. 用最优模型重新估计内群

import numpy as np
import scipy as sp # 科学计算库
import scipy.linalg as linalg # 线性代数库

# 生成数据
def ransac(data, model, n, k, t, d, debug = False, return_all = False):
    """
    输入：
    :param data: 数据点
    :param model: 假设模型:事先自己确定
    :param n: 生成模型所需的最少数据个数
    :param k: 最大迭代次数
    :param t: 阈值:作为判断点满足模型的条件
    :param d: 拟合较好时，需要的样本点最少的个数，当做阈值看待
    :param debug: 是否打印debug信息
    :param return_all: 是否返回所有的符合条件的点
    :return: 最优拟合解（返回nil，如果未找到）

    iterations = 0
    bestfit = nil #记录最优模型的参数估计和内点数目，后面会更新
    besterr = something really large #无穷大 后面会更新
    while iterations < k
    {
        maybeinliers = 从数据集中随机选取n个点,不一定全是内点,也有可能是外点
        maybemodel = n个随机选取的点所建立的模型,比如直线
        alsoinliers = emptyset #初始化内点集合
        for (每一个数据集中不属于maybeinliers的点)
        {
            if 满足maybemodel即错误小于t
                将点加入alsoinliers
        }
        if (alsoinliers样本点数目 > d)
        {
            %有了较好的模型,测试模型符合度
            bettermodel = 利用所有的maybeinliers和alsoinliers重新建立更好的模型
            thiserr = 所有的maybeinliers和alsoinliers的误差度量
            if thiserr < besterr
            {
                bestfit = bettermodel
                besterr = thiserr
            }
        }
        iterations++
    }
    return bestfit
    """

    iterations = 0
    bestfit = None
    besterr = np.inf # 无穷大,后面会更新
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0]) # 随机选取n个点,不一定全是内点,也有可能是外点
        print('test_idxs',test_idxs)
        maybe_inliers = data[maybe_idxs, :] # maybe_idxs是内点的索引,从data中取出内点,作为maybe_inliers
        test_points = data[test_idxs] # test_idxs是外点的索引,从data中取出外点,作为test_points
        maybemodel = model.fit(maybe_inliers) # 用内点去拟合模型,比如用直线去拟合
        test_err = model.get_error(test_points, maybemodel) # 用外点去测试模型,计算误差
        print('test_err',test_err < t) # 打印误差是否小于阈值t
        also_idxs = test_idxs[test_err < t] # 记录误差小于阈值t的点的索引,也就是内点的索引
        print('also_idxs',also_idxs) # 打印内点的索引
        also_inliers = data[also_idxs, :] # 根据索引取出内点
        if debug:
            print ('test_err.min()',test_err.min()) # 打印最小误差
            print ('test_err.max()',test_err.max()) # 打印最大误差
            print ('numpy.mean(test_err)',np.mean(test_err)) # 打印平均误差
            print ('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers))) # 打印内点个数
        # if len(also_inliers) > d: # 如果内点个数大于阈值d,则认为模型已经拟合的比较好
        print('d = ',d) # 打印阈值d 1.5*4
        if len(also_inliers) > d: # 如果内点个数大于阈值d,则认为模型已经拟合的比较好
            betterdata = np.concatenate((maybe_inliers, also_inliers)) # 将内点和外点合并
            bettermodel = model.fit(betterdata) # 利用所有的内点和外点重新建立更好的模型
            better_errs = model.get_error(betterdata, bettermodel) # 计算误差
            thiserr = np.mean(better_errs) # 计算平均误差
            if thiserr < besterr: # 如果平均误差小于besterr,则更新bestfit,besterr,best_inlier_idxs
                bestfit = bettermodel # 更新bestfit
                besterr = thiserr # 更新besterr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs)) # 更新best_inlier_idxs,将新点加入到内点集合中
        iterations += 1 # 迭代次数加1
    if bestfit is None: # 如果迭代次数超过最大迭代次数k,则返回nil
        raise ValueError("did't meet fit acceptance criteria") # 抛出异常
    if return_all: # 如果return_all为True,则返回所有的内点,模型参数,误差
        return bestfit, {'inliers': best_inlier_idxs} # 返回模型参数,内点索引
    else: # 如果return_all为False,则返回模型参数
        return bestfit # 返回模型参数

# 随机分割数据
def random_partition(n, n_data):
    """return n random rows of data and the other len(data)-n rows"""
    all_idxs = np.arange(n_data) # 生成0~n_data-1的数组
    np.random.shuffle(all_idxs) # 打乱数组
    idxs1 = all_idxs[:n] # 取前n个
    idxs2 = all_idxs[n:] # 取后面所有的
    return idxs1, idxs2 # 返回前n个和后面所有的索引

# 最小二乘求解模型参数
class LinearLeastSquareModel:
    # 最小二乘求线性解，用于 RANSAC 的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns # 输入列
        self.output_columns = output_columns # 输出列
        self.debug = debug # 是否打印debug信息

    # 拟合模型
    def fit(self, data):
        # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T # 取出输入列
        B = np.vstack([data[:, i] for i in self.output_columns]).T # 取出输出列
        x, resids, rank, s = np.linalg.lstsq(A, B) # 最小二乘求解
        return x # 返回最小平方和向量

    # 计算误差
    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T # 取出输入列
        B = np.vstack([data[:, i] for i in self.output_columns]).T # 取出输出列
        B_fit = np.dot(A, model) # 计算拟合的输出列, B_fit = A * model.k + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1) # 计算每个点的误差
        return err_per_point # 返回每个点的误差

# 用于测试的数据
def test():
    # 生成理想数据
    n_samples = 500 # 样本数
    n_inputs = 1 # 输入列数，输入变量个数
    n_outputs = 1 # 输出列数，输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs)) # 生成输入列，随机生成0-20之间的500个数据：500*1，np.random.random((n_samples, n_inputs))生成500*1的矩阵，每个元素都是0-1之间的随机数，乘以20之后，每个元素都是0-20之间的随机数
    # np.random.random和np.random在生成随机数的分布类型上有所不同，前者只能生成服从均匀分布的随机数，而后者可以根据不同的分布类型生成符合特定概率分布的随机数
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs)) # 生成输出列，随机生成500个数据：1*1
    B_exact = np.dot(A_exact, perfect_fit) # 生成输出列，500*1

    # 加入高斯噪声,最小二乘能很好的处理高斯噪声
    A_noisy = A_exact + np.random.normal(size=A_exact.shape) # 加入噪声的输入列
    B_noisy = B_exact + np.random.normal(size=B_exact.shape) # 加入噪声的输出列

    # 添加异常值
    if 1:
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0]) # 生成0~499的数组，A_noisy.shape[0]为500
        np.random.shuffle(all_idxs) # 打乱数组，将数组中的元素随机打乱
        outlier_idxs = all_idxs[:n_outliers] # 取前100个，作为异常值
        A_noisy[outlier_idxs] = 20 * np.random.normal(size=(n_outliers, n_inputs)) # 将前100个数据替换成新的随机数
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs)) # 将前100个数据替换成新的随机数
    #setup model
    all_data = np.hstack((A_noisy, B_noisy)) # 将输入列和输出列合并成一个矩阵，500*2
    input_columns = range(n_inputs) # 输入列的索引
    output_columns = [n_inputs + i for i in range(n_outputs)] # 输出列的索引
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug) # 创建模型,用最小二乘求解模型参数

    linear_fit,resids,rank,s = sp.linalg.lstsq(all_data[:,input_columns], all_data[:,output_columns]) # 最小二乘求解模型参数

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug = debug, return_all = True) # 50:最小样本数,1000:最大迭代次数,7e3:内点阈值,300:最大内点数

    if 1:
        import pylab # 导入绘图库
        sort_idxs = np.argsort(A_exact[:, 0]) # 对输入列进行排序
        A_col0_sorted = A_exact[sort_idxs] # 排序后的输入列,秩为2的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data') # 绘制散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label='RANSAC data') # 绘制内点，蓝色
        else:
            pylab.plot(A_noisy[non_outlier_idx, 0], B_noisy[non_outlier_idx, 0], 'k.', label='noisy data') # 绘制散点图
            pylab.plot(A_noisy[outlier_idx, 0], B_noisy[outlier_idx, 0], 'r.', label='outlier data') # 绘制异常值，红色

        pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit') # 绘制拟合曲线
        pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], label='exact system') # 绘制理想曲线
        pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:, 0], label='linear fit') # 绘制最小二乘拟合曲线
        pylab.legend() # 绘制图例
        pylab.show() # 显示图像

if __name__ == '__main__':
    test()
