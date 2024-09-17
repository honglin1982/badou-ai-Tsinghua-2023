#-------------------------------------------------------------#
#   MobileNet的网络部分
#-------------------------------------------------------------#
import warnings                                                         # 引入警告模块
import numpy as np

from keras.preprocessing import image                                   # 引入图像预处理模块

from keras.models import Model                                          # 引入模型模块
from keras.layers import DepthwiseConv2D,Input,Activation,Dropout,Reshape,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D     # 引入各种层模块,用于搭建网络,如卷积层,全局池化层,激活层,批标准化层等
from keras.applications.imagenet_utils import decode_predictions        # 引入预训练模型,用于图像分类
from keras import backend as K                                          # 引入后端模块,用于处理图像数据格式和维度顺序,以及卷积等运算


def MobileNet(input_shape=[224,224,3],                                  # 定义MobileNet网络结构
              depth_multiplier=1,                                       # 深度乘数
              dropout=1e-3,                                             # dropout
              classes=1000):                                            # 分类数

    img_input = Input(shape=input_shape)

    # 224,224,3 -> 112,112,32                                           # 224,224,3 -> 112,112,32
    x = _conv_block(img_input, 32, strides=(2, 2))                      # 自定义卷积块

    # 112,112,32 -> 112,112,64                                          # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)      # 自定义深度卷积块

    # 112,112,64 -> 56,56,128                                           # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier,
                              strides=(2, 2), block_id=2)
    # 56,56,128 -> 56,56,128                                            # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)     # 目的是为了增加网络深度，提高网络的表达能力

    # 56,56,128 -> 28,28,256                                            # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier,
                              strides=(2, 2), block_id=4)
    
    # 28,28,256 -> 28,28,256                                            # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)     # 目的是为了增加网络深度，提高网络的表达能力

    # 28,28,256 -> 14,14,512                                            # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier,
                              strides=(2, 2), block_id=6)
    
    # 14,14,512 -> 14,14,512                                            # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024                                             # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 7,7,1024 -> 1,1,1024                                              # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)                                     # 全局平均池化
    x = Reshape((1, 1, 1024), name='reshape_1')(x)                      # 重塑，将数据变成一维
    x = Dropout(dropout, name='dropout')(x)                             # dropout，防止过拟合
    x = Conv2D(classes, (1, 1),padding='same', name='conv_preds')(x)    # 卷积，用于分类
    x = Activation('softmax', name='act_softmax')(x)                    # 激活函数
    x = Reshape((classes,), name='reshape_2')(x)                        # 重塑，将数据变成一维，用于分类

    inputs = img_input                                                  # 输入

    model = Model(inputs, x, name='mobilenet_1_0_224_tf')               # 定义模型，输入和输出,keras函数式模型
    model_name = 'mobilenet_1_0_224_tf.h5'                              # 模型名称
    model.load_weights(model_name)                                      # 加载模型权重

    return model

# 定义卷积块，卷积块由卷积、批量归一化和激活函数组成
def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel,                                         # 卷积，卷积核大小，卷积核个数
               padding='same',                                          # padding方式，same表示输出和输入大小一致
               use_bias=False,                                          # 是否使用偏置
               strides=strides,                                         # 步长
               name='conv1')(inputs)                                    # 名称
    x = BatchNormalization(name='conv1_bn')(x)                          # 批量归一化
    return Activation(relu6, name='conv1_relu')(x)                      # 激活函数


# 定义深度卷积块，深度卷积块由深度卷积和1*1卷积组成
# 目的是为了增加网络深度，提高网络的表达能力
def _depthwise_conv_block(inputs, pointwise_conv_filters,               # 深度卷积，和卷积块类似，只是卷积核个数为1
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3),
                        padding='same',                                 # padding方式
                        depth_multiplier=depth_multiplier,              # 深度乘数
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)           # 深度卷积，%d表示整数，index从1开始

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)          # 批量归一化,x是深度卷积的输出
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),                          # 1*1卷积，卷积核大小为1*1，卷积核个数为pointwise_conv_filters
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)          # 批量归一化
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)      # 激活函数

def relu6(x):
    return K.relu(x, max_value=6)                                       # 定义relu6激活函数，最大值为6

def preprocess_input(x):                                                # 定义预处理函数
    x /= 255.                                                           # 归一化
    x -= 0.5                                                            # 中心化，使得数据分布在[-0.5,0.5]之间
    x *= 2.                                                             # 放大，使得数据分布在[-1,1]之间
    return x                                                            # 返回预处理后的数据

if __name__ == '__main__':
    model = MobileNet(input_shape=(224, 224, 3))

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))              # 加载图片,image.load_img返回一个PIL对象
    x = image.img_to_array(img)                                         # 转换为数组
    x = np.expand_dims(x, axis=0)                                       # 扩展维度
    x = preprocess_input(x)                                             # 预处理
    print('Input image shape:', x.shape)                                # 打印输入图片的大小

    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds,1))  # 只显示top1

