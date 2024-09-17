import numpy as np
from matplotlib import pyplot as plt

def normal1(lst: np.ndarray) -> np.ndarray:
    """[0, 1]归一化

    Args:
        lst (np.ndarray): 随机参数列表

    Returns:
        np.ndarray: 归一化后的随机参数列表
    """
    '''
    单个数据x的公式：x_ = (x - min) / (max - min)
    '''
    return (lst - np.min(lst)) / (np.max(lst) - np.min(lst))

def normal2(lst: np.ndarray) -> np.ndarray:
    """[-1, 1]归一化

    Args:
        lst (np.ndarray): 随机参数列表

    Returns:
        np.ndarray: 归一化后的随机参数列表
    """
    '''
    单个数据x的公式：x_ = (x - mean) / (max - min)
    '''
    mean = np.mean(lst)
    return (lst - mean) / (np.max(lst) - np.min(lst))

def z_score(lst: np.ndarray) -> np.ndarray:
    """标准化

    Args:
        lst (np.ndarray): 随机参数列表

    Returns:
        np.ndarray: 标准化后的随即参数列表
    """
    '''经过处理后的数据均值为0，标准差为1（正态分布）
       单个数据x的公式：x_ = (x - μ) / σ
    '''
    mean = np.mean(lst)
    s = np.std(lst)
    return (lst - mean) / s

if __name__ == "__main__":

    '''遵从正态分布的数组'''
    n = 50
    # 均值为10，标准差为2的数组
    ws = np.random.normal(10, 2, size=(n))

    # 转换成整数，能计数，画图直观
    for i in range(len(ws)):
        ws[i] = int(ws[i])

    # 排序
    ws = np.sort(ws)
    # 计数
    cs = []
    for i in ws:
        c = np.count_nonzero(ws == i)
        cs.append(c)

    # 通过三种方式得到归一化的值

    res1 = normal1(ws)
    res2 = normal2(ws)
    res3 = z_score(ws)
    # print(np.std(res1))

    '''
    画图
    '''

    plt.plot(ws, cs, color='black', label='source')
    plt.plot(res1, cs, color='y', label='normal1')
    plt.plot(res2, cs, color='blue', label='normal2')
    plt.plot(res3, cs, color='r', label='z_score')

    plt.legend()
    plt.show()