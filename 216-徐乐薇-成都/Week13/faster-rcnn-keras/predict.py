from keras.layers import Input
from frcnn import FRCNN 
from PIL import Image

# 本文件用于测试训练好的模型

frcnn = FRCNN()

while True:
    img = input('img/street.jpg')
    try:
        image = Image.open('img/street.jpg')
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image(image)
        r_image.show()
frcnn.close_session()
    
