# -*- coding: utf-8 -*-

import numpy as np
import pylab


def build_data_set():
    x = 20 * np.random.random((500, 1))
    # np.random.normal 从正态（高斯）分布中抽取随机样本。
    a = 60 * np.random.normal(size=(1, 1))
    # y = a*k
    y = np.dot(x, a)
    # 加入高斯噪声
    x_noisy = x + np.random.normal(size=x.shape)
    y_noisy = y + np.random.normal(size=y.shape)

    # 将部分数据修改为噪声
    # 获取索引
    all_idx = np.arange(x_noisy.shape[0])
    # 打乱索引
    np.random.shuffle(all_idx)
    # 截取部分索引
    idx_arr = all_idx[:100]
    x_noisy[idx_arr] = 20 * np.random.random((100, 1))
    y_noisy[idx_arr] = 50 * np.random.normal(size=(100, 1))
    # np.hstack 将参数元组的元素数组按水平方向进行叠加
    # [[1,2]
    # [2,4]
    # ...]
    return x_noisy, y_noisy, x, y, a


if __name__ == '__main__':
    # 准备数据集
    x_noisy, y_noisy, x_exact, y_exact, a = build_data_set()
    data_set = np.hstack((x_noisy, y_noisy))
    # 作为判断满足模型的阈值
    min_per_err = 7000
    # 迭代次数
    iterate = 1000
    # 生成模型需要的最少样本数量
    min_sample = 50
    # 满足最小平方和条件的样本数量
    min_best_simple = 300

    # 最优拟合解
    best_fit = None
    best_err = np.inf
    best_inliner_idx = None
    for i in range(iterate):
        # 1. 在数据中随机选择几个点设定为内群
        all_idx = np.arange(data_set.shape[0])
        np.random.shuffle(all_idx)
        # 内群点
        inner_idx_arr = all_idx[:min_sample]
        outer_idx_arr = all_idx[min_sample:]
        inner_data = data_set[inner_idx_arr]
        outer_data = data_set[outer_idx_arr]
        # 2. 计算内群模型
        x = np.vstack([inner_data[:, 0]]).T  # 第一列Xi-->行Xi
        y = np.vstack([inner_data[:, 1]]).T  # 第二列Yi-->行Yi
        result = np.linalg.lstsq(x, y, rcond=None)
        model = result[0]
        # 3. 把其它刚才没选到的点带入刚才建立的模型中，计算是否为内群
        # 计算最小误差
        x = np.vstack([outer_data[:, 0]]).T  # 第一列Xi-->行Xi
        y = np.vstack([outer_data[:, 1]]).T  # 第二列Yi-->行Yi
        B_fit = np.dot(x, model)
        err_per_point = np.sum((y - B_fit) ** 2, axis=1)
        also_idx = outer_idx_arr[err_per_point < min_per_err]
        also_data = data_set[also_idx, :]
        if len(also_data) > min_best_simple:
            # 将内群和外群合并
            betterdata = np.concatenate((inner_data, also_data))
            # 计算出合并后的模型
            x = np.vstack([inner_data[:, 0]]).T  # 第一列Xi-->行Xi
            y = np.vstack([inner_data[:, 1]]).T  # 第二列Yi-->行Yi
            model = np.linalg.lstsq(x, y, rcond=None)[0]
            # 计算误差
            x = np.vstack([outer_data[:, 0]]).T  # 第一列Xi-->行Xi
            y = np.vstack([outer_data[:, 1]]).T  # 第二列Yi-->行Yi
            B_fit = np.dot(x, model)
            err_per_point = np.sum((y - B_fit) ** 2, axis=1)
            # 平均误差作为新的误差
            current_err = np.mean(err_per_point)
            # 判断误差
            if current_err < best_err:
                # 误差比上次小就赋值
                best_fit = model
                best_err = current_err
                best_inliner_idx = np.concatenate((inner_idx_arr, also_idx))
    print('结果：\n', best_fit, best_err, best_inliner_idx)

    linear_fit = np.linalg.lstsq(data_set[:, [0]], data_set[:, [1]])[0]
    # 展示
    x_sorted = x_exact[np.argsort(data_set[:, 0])]
    pylab.plot(x_noisy[:, 0], y_noisy[:, 0], 'k.', label='data')  # 散点图
    pylab.plot(x_noisy[best_inliner_idx, 0], y_noisy[best_inliner_idx, 0], 'bx', label="RANSAC data")
    pylab.plot(x_sorted[:, 0],
               np.dot(x_sorted, best_fit)[:, 0],
               label='RANSAC fit')
    pylab.plot(x_sorted[:, 0],
               np.dot(x_sorted, a)[:, 0],
               label='exact system')
    pylab.plot(x_sorted[:, 0],
               np.dot(x_sorted, linear_fit)[:, 0],
               label='linear fit')
    pylab.legend()
    pylab.show()
