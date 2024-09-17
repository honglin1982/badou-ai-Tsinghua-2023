"""这段代码的作用是使用加载好的Faster R-CNN模型对给定的图像进行实时目标检测"""
# from keras.layers import Input
from frcnn import FRCNN 
from PIL import Image

frcnn = FRCNN()    #创建一个FRCNN的实例frcnn，该实例已经加载了预训练的Faster R-CNN模型

#try-except-else是一种异常处理机制，先执行try中代码，没有异常执行else，有异常执行except  

try:
    image = Image.open(r"D:\street.jpg")
except:
    print('Open Error! Try again!')
else:
    r_image = frcnn.detect_image(image)  #调用frcnn.detect_image方法，传入Image对象进行目标检测，返回的结果是带有目标框的图像
    r_image.show()   #显示带有目标框的图像
frcnn.close_session()   #关闭会话，释放资源

    
