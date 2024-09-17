import numpy as np
import scipy.linalg
import pylab

# 初始化样本数据: 参数1=样本数量 参数2=输入变量数 参数3=输出变量数 参数4=孤立点数量
def init_data(number, input_number, output_number, outlier_number):
    # 生成原始数据
    A_samples = 20 * np.random.random((number, input_number))  # 样本 x,范围 0~20
    right_slope = 60 * np.random.normal(size=(input_number, output_number))  # 斜率 k, 范围=0~60, 默认 均值0 标准差1
    B_samples = np.dot(A_samples, right_slope)  # 样本 y, 公式 y = x*k
    print(f"原始模型: y={right_slope[0][0]}x")

    # 高斯噪声处理: 每一个(x, y)值加入服从正态分布的高斯随机数
    A_noisy = A_samples + np.random.normal(size=A_samples.shape)
    B_noisy = B_samples + np.random.normal(size=B_samples.shape)

    # 添加孤立点
    if outlier_number:  # 若孤立点数量为0, 则不添加孤立点
        all_indexes = np.arange(number)  # 创建索引数组 0 ~ number-1
        np.random.shuffle(all_indexes)  # 打乱索引
        outlier_indexes = all_indexes[:outlier_number]  # 获取索引数组中的前 孤立点个数 的数据作为孤立点索引
        # 将孤立点对应的样本数据 (x,y) 重新赋值
        A_noisy[outlier_indexes] = 20 * np.random.random((outlier_number, input_number))
        B_noisy[outlier_indexes] = 50 * np.random.normal(size=(outlier_number, output_number))

    # 将数据进行聚合, shape=(500,2)
    data = np.hstack((A_noisy, B_noisy))
    print(f"样本数据格式:{data.shape}")

    return A_samples, data, right_slope   # 返回处理完毕后的样本数据和斜率

# 使用最小二乘法拟合数据
def ls_fit(data):
    a = data[:, input_col]   # x值
    b = data[:, output_col]  # y值
    x, residuals, rank, s = scipy.linalg.lstsq(a, b)
    return x

# 用于将外群数据代入建立的模型中，返回残差平方和
def model_verification(data, model):
    a = data[:, input_col]   # x 值
    b = data[:, output_col]  # y 值
    b_fit = np.dot(a, model)  # 根据模型求出拟合的y值
    # 求残差平方和，按列求和，每一行所有元素的和为一个值
    residuals = np.sum((b-b_fit) ** 2, axis=1)
    return residuals  # 返回残差平方和

# Ransac算法实现函数: 返回最优拟合解和对应的内群点坐标
def ransac_fit(data, n, k, t, d):
    best_fit = None  # 初始化最优拟合解
    best_residuals = np.inf  # 初始化最优误差, np.inf表正无穷大
    best_inlier_index = None  # 初始化最优内群点索引

    # 进行 Ransac 迭代
    for i in range(k):
        all_index = np.arange(data.shape[0])  # 获取所有样本的索引，一维
        np.random.shuffle(all_index)  # 随机所有的索引
        in_index = all_index[:n]    # 内群点索引
        out_index = all_index[n:]   # 外群点的索引
        in_data = data[in_index]    # 内群数据
        out_data = data[out_index]  # 外群数据
        # 通过最小二乘法拟合可能的模型
        maybe_model = ls_fit(in_data)
        # 将外群点代入模型中验证是否为内群点, 返回残差平方和
        residuals = model_verification(out_data, maybe_model)
        # 将残差平方和 与 阈值 t 做比较, 小于阈值的即为新的内群点
        new_in_index = out_index[residuals < t]  # 获取比较结果为 True 的外群索引
        new_in_data = data[new_in_index, :]  # 根据索引获取对应的新加内群点数据,
        # print(f"第{i+1}次迭代,残差平方和的最小值:{residuals.min()} 最大值:{residuals.max()} 平均值:{np.mean(residuals)} 小于阈值{t}的外群点数量:{len(new_in_data)}")

        # 拟合较好时,需要的样本点最少的个数,当做阈值看待
        if len(new_in_data) > d:
            # 更新后的内群数据
            better_data = np.concatenate((in_data, new_in_data))
            better_model = ls_fit(better_data)  # 新的拟合模型
            better_residuals = model_verification(better_data, better_model)  # 新的残差平方和
            mean_residuals = np.mean(better_residuals)  # 将平均残差平方和作为新的残差平方和
            # 通过比较残差平方和的大小，来决定是否更新拟合结果
            if mean_residuals < best_residuals:
                best_fit = better_model   # 拟合
                best_residuals = mean_residuals  # 残差平方和
                best_inlier_index = np.concatenate((in_index, new_in_index))  # 最优内群索引集合

    if best_fit is None:
        raise ValueError("没有找到最优拟合解")

    return best_fit, best_inlier_index, best_residuals


# 绘图函数
def drawing(A, data, ransac_index, fit1, fit2, fit3):
    a_noisy = data[:, input_col]
    b_noisy = data[:, output_col]

    sort_index = np.argsort(A[:, 0])
    A_col0_sorted = A[sort_index]  # 秩为2的数组

    # 绘制散点图
    pylab.plot(a_noisy[:, 0], b_noisy[:, 0], 'k.', label='data')
    pylab.plot(a_noisy[ransac_index, 0], b_noisy[ransac_index, 0], 'bx', label="RANSAC data")

    # 绘制直线图, 使用原始x数据是为了对比 ransac fit 拟合效果与开始 fit 的区别
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, fit1)[:, 0], label='init fit')
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, fit2)[:, 0], label='lstsq fit')
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, fit3)[:, 0], label='Ransac fit')

    pylab.legend()  # 显示标签
    pylab.show()    # 显示图像


if __name__ == '__main__':
    input_var = 1      # 输入变量数
    output_var = 1     # 输出变量数
    input_col = [0]    # 输入变量在样本数据中所在的特征列
    output_col = [1]   # 输出变量在样本数据中所在的特征列
    sample_num = 500   # 样本数
    outlier_num = 100  # 样本孤立点的数量
    n_ = 50    # 生成模型所需的最少样本点
    k_ = 1000  # 最大迭代次数
    t_ = 7000  # 阈值:用于判断点是否满足模型的条件
    d_ = 300   # 拟合较好时,需要的样本点最少的个数,当做阈值看待

    # 返回样本数据与原始模型
    A_init, all_data, fit = init_data(sample_num, input_var, output_var, outlier_num)

    # 返回最小二乘拟合的模型
    lstsq_fit = ls_fit(all_data)
    print(f"最小二乘拟合的模型: y={lstsq_fit[0][0]}x")

    # 返回 Ransac 拟合的模型、内群点索引、残差平方和
    ransac_fit, ransac_inlier_index, best_residuals = ransac_fit(all_data, n_, k_, t_, d_)
    print(f"Ransac拟合的模型: y={ransac_fit[0][0]}x")
    # print(ransac_inlier_index.shape, best_residuals)
    # 绘图
    drawing(A_init, all_data, ransac_inlier_index, fit, lstsq_fit, ransac_fit)