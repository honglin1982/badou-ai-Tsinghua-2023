import numpy as np
import matplotlib.pyplot as plt

def normaliszm(x):
    num_min = min(x)
    num_max = max(x)
    y = np.array([(i-num_min)/(num_max-num_min) for i in x])
    print(f"normaliszm:{y}\n")
    return y
def normalizm0(x):
    average = sum(x)/len(x)
    s = sum([(i-average)**2 for i in x])/len(x)
    s = s**0.5
    y = np.array([(i-average)/s for i in x])
    print(f"normalizm0:{y}\n")
    return y

x = 20*np.random.rand(13)
print(f"x:{x}\n")
print(f"x:{np.nor}\n")
normaliszm(x)
normalizm0(x)


