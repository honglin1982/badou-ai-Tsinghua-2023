import os

photos = os.listdir("./data/image/train/")  #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。

# 该部分用于生成数据集的txt
with open("data/dataset.txt","w") as f:     #with open(,w) 以写入的方式打开文件
    for photo in photos:
        name = photo.split(".")[0]          #split() 通过指定分隔符对字符串进行切片
        if name=="cat":                     #split(".") 以"."为分隔符
            f.write(photo + ";0\n")
        elif name=="dog":
            f.write(photo + ";1\n")
f.close()