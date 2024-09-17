import numpy as np

from nets.resnet50 import resnet50

# from keras.preprocessing import image
from keras.utils import image_utils

from keras.applications.imagenet_utils import preprocess_input, decode_predictions


if __name__ == '__main__':
    model = resnet50(input_shape=[224, 224, 3])
    model.summary()

    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    img = image_utils.load_img(img_path, target_size=(224, 224))
    # print(img.size)
    img_x = image_utils.img_to_array(img)
    print(f'---img_x: {img_x.shape}')
    img_x = np.expand_dims(img_x, axis=0)  # 增加维度
    print(f'---img_x: {img_x.shape}')
    x = preprocess_input(img_x)  # 类似于一个归一化的函数
    print(f'---x: {x.shape}')
    preds = model.predict(x)
    # print(f'--pres: {preds}')  # softmax 的概率
    print('Predicted:', decode_predictions(preds))  # 解码类似
    pass
