from __future__ import print_function  #在代码中使用print()函数而不是print语句
import numpy as np
from keras import layers   #用于构建深度神经网络的各种层

from keras.layers import Input
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Activation,BatchNormalization,Flatten  #用于添加激活函数，批归一化和展平层
from keras.models import Model  #用于构建模型

from keras.preprocessing import image  #用于处理图像数据
import keras.backend as K    #进行底层的张量操作
from keras.utils.data_utils import get_file    #用于从网络上下载数据文件
from keras.applications.imagenet_utils import decode_predictions   #用于将模型的输出解码为类别标签
from keras.applications.imagenet_utils import preprocess_input   #用于对输入进行预处理，使其适应模型的要求

#构建ResNet50模型中的恒等块（identity block）
def indentity_block(input_tensor,kernel_size,filters,stage,block):
    filters1,filters2,filters3=filters
    Conv2Dconv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x=Conv2D(filters1,(1,1),name=Conv2Dconv_name_base+'2a')(input_tensor)
    x=BatchNormalization(name=bn_name_base+'2a')(x)
    x=Activation('relu')(x)

    x=Conv2D(filters2,kernel_size,padding='same',name=Conv2Dconv_name_base+'2b')(x)
    x=BatchNormalization(name=bn_name_base+'2b')(x)
    x=Activation('relu')(x)

    x=Conv2D(filters3,(1,1),name=Conv2Dconv_name_base+'2c')(x)
    x=BatchNormalization(name=bn_name_base+'2c')(x)

    x=layers.add([x,input_tensor])   #引入输入数据
    x=Activation('relu')(x)
    return x

#构建ResNet50模型中的卷积块（convolution block）
# 参数分别为输入张量，卷积核大小，滤波器数量，阶段，块编号，步长
def conv_block(input_tensor,kernel_size,filters,stage,block,strides=(2,2)):
    filters1,filters2,filters3=filters   #分别表示三个卷积层的卷积核数量
    conv_name_base='res'+str(stage)+block+'_branch'  #卷积层名称
    bn_name_base='bn'+str(stage)+block+'_branch'     #批归一化名称
    
    x=Conv2D(filters1,(1,1),strides=strides,name=conv_name_base+'2a')(input_tensor)
    x=BatchNormalization(name=bn_name_base+'2a')(x)
    x=Activation('relu')(x)

    x=Conv2D(filters2,kernel_size,padding='same',name=conv_name_base+'2b')(x)
    x=BatchNormalization(name=bn_name_base+'2b')(x)
    x=Activation('relu')(x)

    x=Conv2D(filters3,(1,1),name=conv_name_base+'2c')(x)
    x=BatchNormalization(name=bn_name_base+'2c')(x)       #主路径

    shortcut=Conv2D(filters3,(1,1),strides=strides,name=conv_name_base+'1')(input_tensor)
    shortcut=BatchNormalization(name=bn_name_base+'1')(shortcut)   #残差路径
 
    x=layers.add([x,shortcut])    
    x=Activation('relu')(x)
    return x



#构建ResNet50模型
def ResNet50(input_shape=[224,224,3],classes=1000):
    img_input=Input(shape=input_shape)  #创建一个输入层，用于定义模型的输入
    x=ZeroPadding2D((3,3))(img_input)  #进行零填充

    x=Conv2D(64,(7,7),strides=(2,2),name='conv1')(x)  #卷积
    x=BatchNormalization(name='bn_conv1')(x)   #批归一化
    x=Activation('relu')(x)   #激活函数
    x=MaxPooling2D((3,3),strides=(2,2))(x)  #池化
    #一系列的卷积块和恒等块
    x=conv_block(x,3,[64,64,256],stage=2,block='a',strides=(1,1))
    x=indentity_block(x,3,[64,64,256],stage=2,block='b')
    x=indentity_block(x,3,[64,64,256],stage=2,block='c')

    x=conv_block(x,3,[128,128,512],stage=3,block='a')
    x=indentity_block(x,3,[128,128,512],stage=3,block='b')
    x=indentity_block(x,3,[128,128,512],stage=3,block='c')
    x=indentity_block(x,3,[128,128,512],stage=3,block='d')

    x=conv_block(x,3,[256,256,1024],stage=4,block='a')
    x=indentity_block(x,3,[256,256,1024],stage=4,block='b')
    x=indentity_block(x,3,[256,256,1024],stage=4,block='c')
    x=indentity_block(x,3,[256,256,1024],stage=4,block='d')
    x=indentity_block(x,3,[256,256,1024],stage=4,block='e')
    x=indentity_block(x,3,[256,256,1024],stage=4,block='f')

    x=conv_block(x,3,[512,512,2048],stage=5,block='a')
    x=indentity_block(x,3,[512,512,2048],stage=5,block='b')
    x=indentity_block(x,3,[512,512,2048],stage=5,block='c')

    x=AveragePooling2D((7,7),name='avg_pool')(x)  #平均池化，将特征图的维度将为1，池化前输入（7,7,2048）
    x=Flatten()(x)
    x=Dense(classes,activation='softmax',name='fc1000')(x)   #添加全连接层

    model=Model(img_input,x,name='resnet50')   #使用输入层和输出层创建一个模型对象
    model.load_weights(r"D:\AI\submit assignment\Assignment\week11\resent\resnet50_weights_tf_dim_ordering_tf_kernels.h5")  #加载预训练权重

    return model



#主函数
if __name__=='__main__':
    model=ResNet50()
    model.summary()   #调用summary()方法，打印模型的结构摘要
    img_path=r"D:\AI\submit assignment\Assignment\week11\resent\elephant.jpg"
    img=image.load_img(img_path,target_size=(224,224))  #加载指定路径图片并调整尺寸
    x=image.img_to_array(img)  #转化为数组
    x=np.expand_dims(x,axis=0)  #为数组x增加一个维度
    x=preprocess_input(x)    #数据预处理，适应ResNet50模型要求

    print('Input image shape:',x.shape)  #大于打印图形形状
    preds=model.predict(x)    #对输入数据分类预测
    print('predicted:',decode_predictions(preds))  