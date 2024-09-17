
import numpy as np
import matplotlib.pyplot as plt


def normalize(data):
    """Normalize data to [0, 1] range."""
    data = np.array(data)
    return (data - data.min()) / (data.max() - data.min())

def normalize2(data):
    """Normalize data to [-1, 1] range."""
    data = np.array(data)
    return (data - data.min()) / (data.max() - data.min()) * 2 - 1

def normalize3(data):
    """Normalize data to [0, 1] range."""
    data = np.array(data)
    return (data - data.mean()) / (data.max() - data.min())

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * 3
    plt.plot(x, y, 'go')
    plt.plot(x, normalize(y), 'ro')
    plt.plot(x, normalize2(y), 'bo')
    plt.plot(x, normalize3(y), 'co')
    plt.show()