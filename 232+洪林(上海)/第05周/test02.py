# 有志者事竟成，破釜沉舟，百二秦关终属楚。
# 苦心人天不负，卧薪尝胆，三千越甲可吞吴。
# @File     : test02.py
# @Author   : honglin
# @Time     : 2023/6/3 23:05
zhan = [1,2,3,4,5]
print(zhan)
print("len(zhan) = ", len(zhan))

while not len(zhan) == 0:
   tmp =  zhan.pop()  # 出栈
   # zhan.append(tmp)
   print(tmp)


print(zhan)