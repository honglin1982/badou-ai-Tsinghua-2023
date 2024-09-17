import os

photoNames = os.listdir('./data/image/train')

with open('data/dataset.txt', 'w') as f:
    for pn in photoNames:
        name = pn.split('.')[0]
        f.write(pn + ';' + ('0' if name == 'cat' else '1' if name == 'dog' else '-1') + "\n")
