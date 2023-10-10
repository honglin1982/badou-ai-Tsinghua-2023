#使用预训练的AlexNet模型对输入图像进行分类预测，并显示结果
import numpy as  np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet


#检查并设置keras的图像维度顺序
if K.common.image_dim_ordering() !='tf':
    K.commom.set_image_dim_ordering('tf')

if __name__=='__main__':
    model=AlexNet()
    model.load_weights('./logs/ep039-loss0.004-val_loss0.652.h5') #从外部文件中加载神经网络模型的权重
    img=cv2.imread('D:\AI\submit assignment\Assignment\week11\Alexnet\Test.jpg')
    img_RGB=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    img_nor=img_RGB/255
    img_nor=np.expand_dims(img_nor,axis=0)  #插入一个新的维度
    img_resize=utils.resize_image(img_nor,(224,224))
    print(utils.print_answer(np.argmax(model.predict(img_resize))))  #将类别索引转化为类别标签并输出
    cv2.imshow('ooo',img)
    cv2.waitKey(0)

