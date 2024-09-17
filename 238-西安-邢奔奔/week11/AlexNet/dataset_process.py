import os

# 用于返回指定目录中的所有文件和文件夹的名称列表
photos = os.listdir('./image/train')

with open("./image/dataset.txt", 'w') as f:
    for photo in photos:
        name = photo.split(',')[0]
        if name == "cat":
            f.write(photo + ";0\n")
        elif name == "dog":
            f.write(photo + ";1\n")
f.close()
