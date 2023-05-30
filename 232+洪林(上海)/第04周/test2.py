# 有志者事竟成，破釜沉舟，百二秦关终属楚。
# 苦心人天不负，卧薪尝胆，三千越甲可吞吴。
# @File     : test2.py
# @Author   : honglin
# @Time     : 2023/5/31 0:50
import numpy as np

b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
ind = [0, 2]

print('--------------------------------------')
print(len(ind))
print(range(len(ind)))
print(range(2))

for i in range(2):
    print(i)
print('--------------------------------------')

result = [b[:, ind[i]] for i in range(len(ind))]

print(result)

print([b[:,0]])
print([b[:,1]])
print([b[:,2]])