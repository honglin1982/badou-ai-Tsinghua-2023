from model import AlexNet_Model
import os
import cv2
import numpy as np


if __name__ == '__main__':
    model = AlexNet_Model(output_shape=2)
    model.load_weights("./logs/last.h5")

    data_path = "./image/train"

    while 1:
        random_index = np.random.randint(0, 12500)
        random_label = 'dog' if np.random.random() > 0.5 else 'cat'

        random_img = os.path.join(data_path, '{}.{}.jpg'.format(random_label, random_index))

        img = cv2.imread(random_img)

        process_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        process_img = cv2.resize(process_img, (227, 227))
        process_img = process_img / 255.0
        process_img = np.expand_dims(process_img, 0)

        predict = model.predict(process_img)
        predict_label = 'cat' if np.argmax(predict) == 0 else 'dog'

        print("图片：{}，预测结果为{}，预测{}".format(
            random_img,
            predict_label,
            '正确' if random_label == predict_label else '错误'
        ))

        cv2.imshow("img", img)
        cv2.waitKey()