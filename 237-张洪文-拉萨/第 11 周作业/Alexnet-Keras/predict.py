import os
import cv2
import utils
import numpy as np
from tensorflow.keras import backend as K
from model.AlexNet import AlexNet
K.set_image_data_format("channels_last")  # # 设置图像数据的维度顺序,顺序=（h,w,c）

def predict(filename):
    img = cv2.imread(filename)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_rgb / 255
    img_norm = np.expand_dims(img_nor, axis=0)  # 在轴索引为0的轴上添加一个维度
    img_resize = utils.resize_image(img_norm, size=(224, 224))
    predict_result = model.predict(img_resize)  # 获取预测结果分类概率值
    max_index = np.argmax(predict_result)  # 获取最大概率的索引
    print(utils.print_answer(max_index))  # 打印正确答案

    cv2.imshow(filename, img)
    cv2.waitKey(0)


if __name__ == '__main__':
    model = AlexNet()
    model.load_weights("./logs/ep018-loss0.080-val_loss0.431.h5")
    filename_list = ["./data/image/test/"+i for i in os.listdir("./data/image/test/")]
    print(filename_list)
    filename = input("请输入你要预测的图片路径:")
    predict(filename)
