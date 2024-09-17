# -*- coding: utf-8 -*-

if __name__ == '__main__':
    # 线性回归函数：y = ax + b
    # 数据集
    x = [1, 2, 3, 4]
    y = [6, 5, 7, 10]
    # 最小二乘法:  minΣ(yn-(ax+b))²
    # 分别对a和b求偏导，然后令导数为0，可得极致点
    #     NΣ(xn * yn) - (Σxn)*(Σyn)
    # a = —————————————————————————
    #         NΣ(xn)² - (Σxn)²
    #
    #     Σ(yn)        Σ(xn)
    # b = ————— - a(——————————)
    #       N           N

    N = len(x)  # 记录数

    sum_x = 0  # Σxn
    sum_y = 0  # Σyn
    sum_x_y = 0  # Σ(xn * yn)
    sum_x_square = 0  # Σ(xn)²
    for i in range(N):
        sum_x += x[i]
        sum_y += y[i]
        sum_x_y += x[i] * y[i]
        sum_x_square += x[i] ** 2
    # 计算斜率和截距
    a = ((N * sum_x_y) - (sum_x * sum_y)) / (N * sum_x_square - sum_x**2)
    b = sum_y/N - a*(sum_x / N)
    print('y = {}x + {}'.format(a, b))
