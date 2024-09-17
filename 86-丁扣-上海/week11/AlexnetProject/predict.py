import numpy as np
import utils
import cv2
from keras import backend as K
from keras.models import load_model
from model.AlexNet import AlexNetModel

K.set_image_data_format('channels_last')


'''
model.save()保存了模型的图结构和模型的参数，保存模型的后缀是.hdf5。使用只需要 load_model
model. save_weights ()只保存了模型的参数，并没有保存模型的图结构,保存模型的后缀使用.h5。使用这个 加载的时候 .load_weights前面需要加载网络结构

使用save_weights保存的模型比使用save() 保存的模型的大小要小。同时加载模型时的方法也不同。model.save()保存了模型的图结构，直接使用load_model()方法就可加载模型然后做测试。
加载save_weights保存的模型就稍微复杂了一些，还需要再次描述模型结构信息才能加载模型
'''


if __name__ == '__main__':
    model = AlexNetModel(input_shape=(224, 224, 3), output_shape=2)
    model = model.produce_model()
    model.load_weights(r'./logs/last1.h5')
    # model = load_model(r'./logs/ep039-loss0.004-val_loss0.652.h5')
    # model = load_model(r'./logs/last1.h5')
    # image = cv2.imread(f'./Test.jpg', 0)
    # image = cv2.imread(f'./test_dog.png', 0)
    image = cv2.imread(f'./img.png', 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    image = utils.resize_image(image, (224, 224))
    print(image.shape)
    result = model.predict(image)
    print(f'---result: {result}')
    r = utils.output_answer(int(np.argmax(result)))
    print(r)

    pass

