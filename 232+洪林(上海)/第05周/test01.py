# 有志者事竟成，破釜沉舟，百二秦关终属楚。
# 苦心人天不负，卧薪尝胆，三千越甲可吞吴。
# @File     : test01.py
# @Author   : honglin
# @Time     : 2023/6/2 22:19

dim = 5
tmp = [i-dim/2 for i in range(dim)]  # 生成一个序列
print(tmp)

tmp = [i-dim//2 for i in range(dim)]  # 生成一个序列
print(tmp)

print(dim/2)
print(dim//2)

print(8/3)
print(8//3)
print(8.0/3)
print(8.0//3)