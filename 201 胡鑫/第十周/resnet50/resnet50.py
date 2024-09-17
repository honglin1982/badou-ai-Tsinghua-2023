import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as k


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    创建一个identity_block大分支
    :param input_tensor: 输入张量
    :param kernel_size: 卷积核尺寸
    :param filters: 多个卷积核的个数（tuple）
    :param stage: 大分支序号
    :param block: 与stage构成大分支编号
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 与conv_block一致，只是第二部分不进行任何操作，直接对位元素相加
    x = Conv2D(filters1, (1, 1), name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base+'2b')(x)
    x = BatchNormalization(name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base+'2c')(x)
    x = BatchNormalization(name=bn_name_base+'2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    创建一个conv_block大分支
    :param input_tensor: 输入张量
    :param kernel_size: 卷积核尺寸
    :param filters: 多个卷积核的个数（tuple）
    :param stage: 大分支序号
    :param block: 与stage构成大分支编号
    :param strides: 步长
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 分叉：第一部分：卷积、bn、激活
    # 第一次
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    # 第二次
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base+'2b')(x)
    x = BatchNormalization(name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    # 第三次（没有激活）
    x = Conv2D(filters3, (1, 1), name=conv_name_base+'2c')(x)
    x = BatchNormalization(name=bn_name_base+'2c')(x)

    # 第二部分：卷积、bn
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base+'1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base+'1')(shortcut)

    # 将第一部分第三次的输出与第二部分的输出对位元素相加
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=None, classes=1000):
    if input_shape is None:
        input_shape = [224, 224, 3]

    """函数式API写法"""
    # 用Input函数创建一个输入层，shape表示输入数据的形状
    img_input = Input(shape=input_shape)
    # 上下左右各添加三个零填充
    x = ZeroPadding2D((3, 3))(img_input)
    # 7x7卷积，输出特征层为64，步长2
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    # bn操作，所有输出数据均值接近于0，标准差接近1的正太分布，使其落于激活函数敏感区，避免梯度消失，
    # 加快收敛。可以加快模型收敛速度，并且具有一定的泛化能力，还可以减少dropout的使用
    x = BatchNormalization(name='bn_conv1')(x)
    # 激活、池化
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), (2, 2))(x)

    # 第一个大块：一个conv_block接两个identity_block，输入张量h、w减半
    x = conv_block(x, 3, (64, 64, 256), 2, 'a', strides=(1, 1))
    x = identity_block(x, 3, (64, 64, 256), 2, 'b')
    x = identity_block(x, 3, (64, 64, 256), 2, 'c')

    # 第二个大块：一个conv_block接三个identity_block，输入张量h、w减半
    x = conv_block(x, 3, (128, 128, 512), 3, 'a')
    x = identity_block(x, 3, (128, 128, 512), 3, 'b')
    x = identity_block(x, 3, (128, 128, 512), 3, 'c')
    x = identity_block(x, 3, (128, 128, 512), 3, 'd')

    # 第三个大块：一个conv_block接五个identity_block，输入张量h、w减半
    x = conv_block(x, 3, (256, 256, 1024), 4, 'a')
    x = identity_block(x, 3, (256, 256, 1024), 4, 'b')
    x = identity_block(x, 3, (256, 256, 1024), 4, 'c')
    x = identity_block(x, 3, (256, 256, 1024), 4, 'd')
    x = identity_block(x, 3, (256, 256, 1024), 4, 'e')
    x = identity_block(x, 3, (256, 256, 1024), 4, 'f')

    # 第四个大块：一个conv_block接两个identity_block，输入张量h、w减半
    x = conv_block(x, 3, (512, 512, 2048), 5, 'a')
    x = identity_block(x, 3, (512, 512, 2048), 5, 'b')
    x = identity_block(x, 3, (512, 512, 2048), 5, 'c')

    # 平均池化、拍扁、全连接
    # AveragePooling2D、MaxPooling2D默认步长与池化窗口大小相同
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    # 将输入 img_input 和输出 x 构建成一个模型，并将模型命名为 'resnet50'
    model = Model(img_input, x, name='resnet50')

    # 载入模型参数
    model.load_weights("./resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model


if __name__ == '__main__':
    model = ResNet50()
    # 输出模型的摘要信息
    model.summary()
    img_path = 'bike.jpg'
    # image.load_img() 函数并不执行任何图像处理或颜色空间转换操作
    img = image.load_img(img_path, target_size=(224, 224))
    # 将加载的图像 img 转换为一个多维数组 x，其中存储了图像的像素值。(numpy数组)
    tensor = image.img_to_array(img)
    # 数组 x 在第 0 维进行扩展，即在最前面添加一个新的维度。
    tensor = np.expand_dims(tensor, axis=0)
    # 归一化
    tensor = preprocess_input(tensor)

    print("Input image shape: ", tensor.shape)
    preds = model.predict(tensor)
    print("predicted: ", decode_predictions(preds))