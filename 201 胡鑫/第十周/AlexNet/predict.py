import numpy as np
import utils
import cv2
from keras import backend as k
from model.AlexNet import AlexNet

k.set_image_data_format("channels_last")

if __name__ == '__main__':
    model = AlexNet()
    model.load_weights('./logs/ep036-loss0.000-val_loss0.837.h5')

    img = cv2.imread('./test/5.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_rgb / 255
    img_nor = np.expand_dims(img_nor, axis=0)
    img_resize = utils.resize_image(img_nor, (224, 224))

    print(utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow('1', img)
    cv2.waitKey(0)
