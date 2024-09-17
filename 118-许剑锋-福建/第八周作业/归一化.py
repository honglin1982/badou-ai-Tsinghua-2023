import numpy as np
import cv2
import matplotlib.pyplot as plt

def Normalization1(x):
    mean_val = np.mean(x)
    max_val = np.max(x)
    min_val = np.min(x)
    return [(i - mean_val) / (max_val - min_val) for i in x]

def Normalization2(x):
    max_val = np.max(x)
    min_val = np.min(x)
    return [(i - min_val) / (max_val - min_val) for i in x]


if __name__  == '__main__':
    data = np.random.random_integers(0, 100, (1, 20))
    val1 = Normalization1(data)
    val2 = Normalization2(data)
    print(val1)
    print(val2)
