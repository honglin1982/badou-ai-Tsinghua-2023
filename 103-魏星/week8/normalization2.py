import numpy as np
import matplotlib.pyplot as plt

def sigmoid(arr):
    return 1/(1+np.exp(-arr))

def tanh(arr):
    return (np.exp(arr)-np.exp(-arr))/(np.exp(arr)+np.exp(-arr))

def relu(arr):
    return [max(i,0) for i in arr]

arr_inputs = np.arange(-10,10,0.1)
sigmoid_result = sigmoid(arr_inputs)
tanh_result = tanh(arr_inputs)
relu_relu = relu(arr_inputs)

plt.figure()
plt.rcParams['font.sans-serif']=['SimHei'] ##可以使用中文

plt.subplot(121)
plt.plot(arr_inputs, sigmoid_result,label='sigmoid')
plt.plot(arr_inputs, tanh_result,label='tanh')
plt.xlabel("Inputs")
plt.ylabel("Outputs")
plt.legend(loc=2)

plt.subplot(122)
plt.plot(arr_inputs, relu_relu,label='relu')
plt.legend(loc=2)
plt.show()