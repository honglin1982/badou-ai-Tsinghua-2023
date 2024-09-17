import numpy as np
import utils                        #自己写的工具包
import cv2
from keras import backend as K      #Keras后端,用于设置图像的维度顺序
from model.AlexNet import AlexNet   #自己写的AlexNet模型

K.set_image_dim_ordering('tf')      #设置图像的维度顺序为tf

if __name__ == "__main__":
    model = AlexNet()               #建立AlexNet模型
    model.load_weights("./logs/ep039-loss0.004-val_loss0.652.h5") #加载训练好的模型
    img = cv2.imread("./Test.jpg")                                #读取测试图片
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)                 #将BGR转换成RGB
    img_nor = img_RGB/255                                         #归一化
    img_nor = np.expand_dims(img_nor,axis = 0)                    #扩展维度,因为模型是针对4维数据的,所以需要扩展一维,axis = 0表示在第0维扩展,即在最前面扩展
    img_resize = utils.resize_image(img_nor,(224,224))            #将图片缩放到224*224
    #utils.print_answer(np.argmax(model.predict(img)))
    print(utils.print_answer(np.argmax(model.predict(img_resize))))#预测并打印结果,注意,这里的img_resize是4维数据,所以要用model.predict,而不是model.predict_classes,np.argmax()是求最大值的索引
    cv2.imshow("ooo",img)                                          #显示图片
    cv2.waitKey(0)