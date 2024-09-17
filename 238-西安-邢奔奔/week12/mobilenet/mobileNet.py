from keras.models import Model
import warnings
import numpy as np
from keras.preprocessing import image
from keras.layers import DepthwiseConv2D,Input,Activation,Dropout,Reshape,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K

def relu6(x):
    return K.relu(x,max_value=6)
#将小于零的值置为零，并将大于 max_value 的值截断为 max_value

def conv_block(inputs,filters,kernel_shape=(3,3),strides=(1,1)):
    '''
    包含卷积归一化和激活函数
    :param inputs:
    :param filters:
    :param kernel_shape:
    :param strides:
    :return:
    '''
    x = Conv2D(filters,kernel_shape,padding='same',strides=strides,use_bias=False,name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(relu6,name='conv1_relu')(x)
    return x

def depthwise_conv_block(inputs,point_wise_conv_filters,depth_multipliers=1,strides=(1,1),blockid=1):
    ''''
    point_wise_conv_filters 最后需要利用1x1卷积来逐点卷积，对不同通道进行信息交互，然后输出最终要求的特征数
    depth_multiplier：整数，指定输出通道数相对于输入通道数的倍数。默认为 1，即不进行通道数扩展
    '''
    x = DepthwiseConv2D((3,3),padding='same',
                        depth_multiplier=depth_multipliers,strides=strides,use_bias=False,name='conv_dw%d' % blockid)(inputs)
    x = BatchNormalization(name='condw_%d_bn'%blockid)(x)
    x = Activation(relu6,name='conv_dw%d_relu'%blockid)(x)
    x = Conv2D(point_wise_conv_filters,(1,1),padding='same',use_bias=False,strides=(1,1),name='conv_pw%d'%blockid)(x)
    x = BatchNormalization(name='conv_pw%d_bn'%blockid)(x)
    x = Activation(relu6,name='conv_pw%d_relu'%blockid)(x)
    return x

def MobileNet(input_shape=[224,224,3],depth_multiplier=1,dropout=1e-3,classes=1000):
    img_input=Input(shape=input_shape)
    #根据mobilenet来搭建网络
    x = conv_block(img_input,32,strides=(2,2))

    x = depthwise_conv_block(x,64,depth_multiplier,blockid=1)

    x = depthwise_conv_block(x,128,depth_multiplier,strides=(2,2),blockid=2)

    x = depthwise_conv_block(x,128,depth_multiplier,blockid=3)

    x = depthwise_conv_block(x,256,depth_multiplier,strides=(2,2),blockid=4)

    x= depthwise_conv_block(x,256,depth_multiplier,blockid=5)

    x = depthwise_conv_block(x,512,depth_multiplier,strides=(2,2),blockid=6)

    x = depthwise_conv_block(x,512,depth_multiplier,blockid=7)
    x = depthwise_conv_block(x,512,depth_multiplier,blockid=8)
    x = depthwise_conv_block(x,512,depth_multiplier,blockid=9)
    x = depthwise_conv_block(x,512,depth_multiplier,blockid=10)
    x = depthwise_conv_block(x,512,depth_multiplier,blockid=11)

    x = depthwise_conv_block(x,1024,depth_multiplier,strides=(2,2),blockid=12)

    x = depthwise_conv_block(x,1024,depth_multiplier,strides=(2,2),blockid=13)
    '''
    将全局平均池化后得到的特征图从 (None, 1024) 的形状转换为 (1, 1, 1024) 的形状
    将最终的预测结果从 (1, 1, classes) 的形状转换为 (classes,) 的形状，以便与标签进行比较和评估。'''
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,1024),name='reshape_1')(x)
    x =Dropout(dropout,name='dropout')(x)
    x = Conv2D(classes,(1,1),padding='same',name='conv_preds')(x)
    x = Activation('softmax',name='act_softmax')(x)
    x = Reshape((classes,),name='reshape_2')(x)

    inputs = img_input

    model = Model(inputs,x,name='mobilenet_1')
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    return model

def preprocess_inputs(x):
    x /= 255.
    x -=0.5
    x *= 2.
    return x

if __name__=='__main__':
    model = MobileNet(input_shape=(224,224,3))
    img_path = 'elephant.jpg'
    img = image.load_img(img_path,target_size=(224,224,3))
    img = image.img_to_array(img)
    x = np.expand_dims(img,axis=0)
    x = preprocess_inputs(x)

    preds = model.predict(x)
    print("Predictions:",decode_predictions(preds,1))






