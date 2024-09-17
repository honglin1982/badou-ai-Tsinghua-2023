import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet

from keras import backend as K
K.set_image_data_format('channels_last')  # 设置图像维度顺序为 'tf' nhwc

if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("./logs/ep039-loss0.004-val_loss0.652.h5")
    img = cv2.imread("./Test.jpg")
    image_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = image_RGB/255
    img_nor = np.expand_dims(img_nor,axis=0)
    img_resize = utils.resize_image(img_nor,(224,224))
    p = utils.print_answer(np.argmax(model.predict(img_resize)))
    print(p)
    cv2.imshow(p,img)
    cv2.waitKey(0)