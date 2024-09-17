import numpy as np
import utils
import cv2
from keras import backend as K
K.set_image_data_format('channels_last')

from model.AlexNet import  AlexNet

if __name__ == '__main__':

    model = AlexNet()
    model.load_weights('./log/last.h5') #更改h5py版本为2.10
    img = cv2.imread('./Test.jpg')
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgNor = imgRGB / 255
    imgNor = np.expand_dims(imgNor, axis = 0)
    imgResize = utils.resizeImage(imgNor, (224, 224))
    print(utils.printAnswer(np.argmax(model.predict(imgResize))))
    cv2.imshow("111", img)
    cv2.waitKey(0)