import numpy as np
import scipy as sp
import scipy.linalg as sl
import matplotlib.pyplot as plt

# 先布局好ransac的核心运行逻辑
def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    data:数据集
    model:数据模型，首次输入要自行确定
    n:生成模型所需的最少样本点
    k:最大迭代次数
    t:误差阈值，判断点是否满足模型的条件
    d:点数量阈值，拟合模型的最少样本点数量。如果超过d了，要重新拟合
    debug:是否输出所有调试信息
    return_all:是否输出所有模型参数

    该函数return最佳拟合的模型参数bestfit,如果未找到返回nil
    """

    # 设置初始默认值：迭代次数、最优拟合解、最优误差、最优内群点的索引：
    iterations = 0
    bestfit = None
    besterr = np.inf       #np.inf作用是初始设为较大实数值，会根据数据集不断更新为更小值
    best_inlier_idxs = None

    # 开始迭代
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0]) #内群索引、外群索引
        print('test_idxs:', test_idxs)

        maybe_inliers = data[maybe_idxs, :] # 从data里取内群点
        test_points = data[test_idxs]  # 取外群点

        maybemodel = model.fit(maybe_inliers)
        test_err = model.get_error(test_points, maybemodel)
        print('test_err=', test_err < t)

        also_idxs = test_idxs[test_err < t]  # 布尔类型数组在test_idxs里找到test_err < t的部分。就是外群里满足条件的点索引
        print('also_idxs=', also_idxs)

        also_inliers = data[also_idxs, :]

        # 调试时打印一些信息，以便调试和优化
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(also_inliers) = %d'%(iterations, len(also_inliers)))
        print('d = ', d)

        # 当外群内样本点数量大于d时，重新拟合模型，得出结果bestfit,besterr,best_inlier_idxs
        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            bettermodel = model.fit(betterdata)
            bettererror = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(bettererror)
            if thiserr < test_err:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs)) # 更新内群索引值,将新点加入

        iterations += 1

    # 出循环总结情况：
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers':best_inlier_idxs}
    else:
        return bestfit

# 工具函数1：获得随机数据集（索引值）的n个值
def random_partition(n, n_data):
    all_idxs = np.arange(n_data) # 获取数据集的下标索引
    np.random.shuffle(all_idxs) # 打乱此下标索引
    idxs1 = all_idxs[:n] 
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

# 工具2：封装最小二乘法模型的求解过程。内置fit方法和get_error方法
class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug = False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
        # input_colomns, output_colomns是指定输入变量和输出变量的列，帮助正确提取数据

    def fit(self, data):
        A = np.vstack( [data[:i] for i in self.input_columns] ).T # 第一列Xi-->行Xi
        B = np.vstack( [data[:i] for i in self.output_columns] ).T # 第二列Yi-->行Yi
        # 对AB进行线性最小二乘拟合，得到最小平方解x，以及残差和resids、秩rank和奇异值s
        x, resids, rank, s = sl.lstsq(A, B)
        return x

    def get_error(self, data, model):
        A = np.vstack( [data[:i] for i in self.input_columns] ).T
        B = np.vstack( [data[:i] for i in self.output_columns] ).T
        B_fit = np.dot(A, model) #矩阵乘法函数，A是输入 * 模型 = 预测输出值
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point

# 测试环节：生成理想数据、建立模型、运行ransac
def test():
    # 生成理想数据，线性回归模型中输入数据满足高斯分布，参数k是独立同分布的随机变量，也满足高斯分布
    n_samples = 500 #样本个数
    n_inputs = 1 #输入变量个数
    n_outputs = 1 #输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  #随机生成500个0-20间数据，行向量
    perfect_fit = 60 * np.random.normal(size = (n_inputs, n_outputs)) #随机生成满足高斯分布的线性度（斜率k）
    B_exact = np.dot(A_exact, perfect_fit)  # y = x*k

    # 加入高斯噪声，使最小二乘法更好处理
    A_noisy = A_exact + np.random.normal(size = A_exact.shape)  # Xi
    B_noisy = B_exact + np.random.normal(size = B_exact.shape)  # Yi

    #手动再添加局外点，该操作可增可减具备可调节性
    if 1:  
        n_outliers = 100  # 假设要添加100个点
        all_idxs = np.arange( A_noisy.shape[0] )   # 获取输入数据的0-499索引
        np.random.shuffle(all_idxs)  # 把这500个索引打乱
        outliers_idxs = all_idxs[:n_outliers]   #取100个作为局外点的索引

        # 生成加入局外点的Xi Yi，Xi用了均匀分布，Yi用了高斯分布，可以模拟更多异常，增强模型鲁棒性
        # 选择噪声分布的方式取决于具体的应用场景和任务需求，可以根据需要选择均匀分布、正态分布或其他分布。
        A_noisy[outliers_idxs] = 20 * np.random.random((n_outliers, n_inputs))
        B_noisy[outliers_idxs] = 50 * np.random.normal(size = (n_outliers, n_outputs))

    # 建立模型：
    all_data = np.hstack((A_noisy, B_noisy))  #[Xi,Yi] 共500行
    input_columns = range(n_inputs) # n_inputs=1所以是索引0，第一列
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 表达是最后一列，但也是第一列
    debug = False
    ## 类的实例化 （上述工具2 最小二乘）
    model = LinearLeastSquareModel(input_columns, output_columns, debug = debug)
    linear_fit,resids,rank,s = sp.linalg.lstsq(all_data[:,input_columns], all_data[:,output_columns])

    # 运行RANSAC算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug = debug, return_all = True)

    if 1:
        sort_idxs = np.argsort(A_exact[:, 0])  # 对A_exact的第一列（样本）进行排序，并返回排序后的索引值
        A_col0_sorted = A_exact[sort_idxs]   # 按照样本的顺序对A_exact进行重排

        if 1:
            plt.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label = 'data')
            # A_noisy数组的第一列作为x轴坐标,B_noisy[:, 0]表示将B_数组的第一列作为y轴坐标,k.是黑色圆点，data是这幅散点图的图名
            plt.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label='ransac data')
            # 认定内群点的分布图，bx是蓝色叉号
        else:
            plt.plot(A_noisy[non_outlier_idxs,0], B_noisy[non_outlier_idxs,0], 'k.', label='noisy data')
            # 认定非离群值的散点图
            plt.plot(A_noisy[outlier_idxs,0], B_noisy[outlier_idxs,0], 'r.', label='outlier data')
            # 认定离群值的散点图，r是红色圆点

        plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted,ransac_fit)[:,0], label='RANSAC fit')
        # ransac模型拟合的曲线图：A_col0_sorted是横坐标，np.dot是乘ransac得出的参数作为纵坐标
        plt.plot( A_col0_sorted[:,0], np.dot(A_col0_sorted,perfect_fit)[:,0], label='exact system' )
        # 精确模型拟合的曲线图
        plt.plot( A_col0_sorted[:,0], np.dot(A_col0_sorted,linear_fit)[:,0], label='linear fit' )
        # 线性模型拟合的曲线图

        plt.legend()  # 添加图例
        plt.show()  # 显示绘图结果


# 主函数：运行test部分
if __name__ == "__main__":
    test()

