import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet

#K.set_image_dim_ordering('tf')
K.image_data_format() == 'channels_first' #chw/hwc

if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("./logs/last1.h5")
    img = cv2.imread("./Test.jpg")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255 #灰度化
    img_nor = np.expand_dims(img_nor,axis = 0) #预处理
    img_resize = utils.resize_image(img_nor,(224,224)) #resize
    #utils.print_answer(np.argmax(model.predict(img)))
    print('the answer is: ', utils.print_answer(np.argmax(model.predict(img_resize))))
	#print(img_size.shape[0])
	#print(img_size.shape[1])
	#print(img_size.shape[2])
    cv2.imshow("ooo",img)
    cv2.waitKey(0)