import cv2
import numpy as np

with open(r"../data/dataset.txt", "r") as f:
    lines = f.readlines()
num_train = int(len(lines)*0.9)
batch_size = 128
i = 0
for b in range(batch_size):
    if i == 0:
        np.random.shuffle(lines[:num_train])
    name = lines[i].split(';')[0]
    img = cv2.imread(r"..\\data\image\train" + '/' + name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
