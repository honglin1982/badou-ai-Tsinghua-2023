import numpy as np
import matplotlib.pyplot as plt


def normalization1(x):
    return [(i - min(x)) / (max(x) - min(x)) for i in x]


def z_score(x):
    avg = np.mean(x, axis=0)
    s = np.sqrt(sum([(i - avg) ** 2 for i in x]) / len(x))
    return [(i - avg) / s for i in x]


l = np.random.randint(0, 10, (10, 1))
print(normalization1(l))
print(z_score(l))
