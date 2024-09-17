import numpy as np
import matplotlib.pyplot as plt

def NormalizationZeroOne(data):
    #归一化范围[0, 1]
    mn = min(data)
    sub = max(data) - min(data)
    return [(float(x) - mn) / sub for x in data]

def NormalizaionMinusOneToOne(data):
    #归一化范围[-1,1]
    sub = max(data) - min(data)
    mean = np.mean(data)
    return [(x - mean) / sub for x in data]

def Zeros(data):

    mean = np.mean(data)
    s2 = sum((mean - x) * (mean - x) for x in data) / len(data)
    return [(x - mean) / s2 for x in data]

def test(data):

    plt.rcParams['font.sans-serif'] =['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sY = [data.count(i) for i in data]
    n1 = NormalizationZeroOne(data)
    n2 = NormalizaionMinusOneToOne(data)
    z = Zeros(data)
    print(n2)
    print(z)
    plt.plot(data, sY, label='原始数据')
    plt.plot(n1, sY, label='0~1范围标准化')
    plt.plot(n2, sY, label='-1~1范围标准化')
    plt.plot(z, sY, label='正态分布标准化')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5,
    #  5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 1]
    # [-0.5, -0.125, -0.125, -0.1, -0.1, -0.1, -0.075, -0.075, -0.075, -0.075, -0.05, -0.05, -0.05, -0.05, -0.05, -0.025,
    #  -0.025, -0.025, -0.025, -0.025, -0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 0.025, 0.025, 0.025, 0.025,
    #  0.025, 0.05, 0.05, 0.05, 0.05, 0.05, 0.075, 0.075, 0.075, 0.075, 0.1, 0.1, 0.1, 0.125, 0.125, 0.5]
    # [-0.875, -0.21875, -0.21875, -0.175, -0.175, -0.175, -0.13125, -0.13125, -0.13125, -0.13125, -0.0875, -0.0875,
    #  -0.0875, -0.0875, -0.0875, -0.04375, -0.04375, -0.04375, -0.04375, -0.04375, -0.04375, 0.0, 0.0, 0.0, 0.0, 0.0,
    #  0.0, 0.0, 0.04375, 0.04375, 0.04375, 0.04375, 0.04375, 0.04375, 0.0875, 0.0875, 0.0875, 0.0875, 0.0875, 0.13125,
    #  0.13125, 0.13125, 0.13125, 0.175, 0.175, 0.175, 0.21875, 0.21875, 0.875]

    test([-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30])