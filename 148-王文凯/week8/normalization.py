import numpy as np
import matplotlib.pyplot as plt

def classic_normalization(x):
    # y = (x - min) / (max - min)
    max_x, min_x = max(x), min(x)
    return [((float(i) - min_x ) / (max_x - min_x)) for i in x]

def zero_mean_normalization(x):
    # y = (x - μ) / σ
    x_mean = np.mean(x)
    s = sum([((i - x_mean) * (i - x_mean)) for i in x]) / len(x)
    return [((i - x_mean) / s) for i in x]

def main():
    test_data = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
         11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    cs = []
    for i in test_data:
        c = test_data.count(i)
        cs.append(c)
    n, z = classic_normalization(test_data), zero_mean_normalization(test_data)
    plt.plot(test_data, cs)
    plt.plot(n, cs)
    plt.plot(z, cs)
    plt.show()

if __name__ == "__main__":
    main()
