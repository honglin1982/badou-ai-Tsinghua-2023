#这段代码是一个目标检测任务的数据预处理和损失函数定义脚本，
#它可以用于生成训练数据和计算损失，从而用于训练目标检测模型。

#导入依赖库，Keras(用于构建和训练模型)，TensorFlow(后端计算库)，numpy(数值计算)，PIL(用于图像处理)
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
import keras
import tensorflow as tf
import numpy as np
from random import shuffle
import random
from PIL import Image
from keras.objectives import categorical_crossentropy
from keras.utils.data_utils import get_file
from matplotlib.colors import rgb_to_hsv,hsv_to_rgb
from utils.anchors import get_anchors
import time

#辅助函数定义
#生成一个在[a,b)范围的随机数
def rand(a=0,b=1):
    return np.random.rand()*(b-a)+a

#损失函数定义
#计算分类损失，y_true真实标签, y_pred预测结果，y_true=[batch_size, num_anchor, num_classes+1]（[批量大小，锚框的数量，目标类别的数量+1]）
def cls_loss(ratio=3):  #参数ratio将用于加权分类损失
    #内部函数_cls_loss计算了一个目标检测模型中的分类损失
    def _cls_loss(y_true,y_pred):
        labels=y_true
        anchor_state=y_true[:,:,-1] #提取每个锚框的状态标签(0背景，1存在目标)，-1代表的是都包含了
        classification=y_pred
        #找出存在目标的先验框
        indices_for_object=tf.where(keras.backend.equal(anchor_state,1))  #找出存在目标的锚框的索引
        labels_for_object=tf.gather_nd(labels,indices_for_object)  #使用找到的索引从labels中获取存在目标的标签数据
        classsification_for_object=tf.gather_nd(classification,indices_for_object) #使用找到的索引从classifications中获取存在目标的分类预测
        cls_loss_for_object=keras.backend.binary_crossentropy(labels_for_object,classsification_for_object)#计算存在目标的锚框的二元交叉熵分类损失
        #找出实际为背景的先验框
        indices_for_back=tf.where(keras.backend.equal(anchor_state,0))
        labels_for_back=tf.gather_nd(labels,indices_for_back)
        classsification_for_back=tf.gather_nd(classification,indices_for_back)
        cls_loss_for_back=keras.backend.binary_crossentropy(labels_for_back,classsification_for_back)
        #分别计算正负样本的数量
        #normalizer_pos代表正样本的数量，经过操作，确保其值为一个浮点数，同时至少为1.0
        normalizer_pos=tf.where(keras.backend.equal(anchor_state,1))
        normalizer_pos=keras.backend.cast(keras.backend.shape(normalizer_pos)[0],keras.backend.floatx())
        normalizer_pos=keras.backend.maximum(keras.backend.cast_to_floatx(1.0),normalizer_pos)

        normalizer_neg=tf.where(keras.backend.equal(anchor_state,0))
        normalizer_neg=keras.backend.cast(keras.backend.shape(normalizer_neg)[0],keras.backend.floatx())
        normalizer_neg=keras.backend.maximum(keras.backend.cast_to_floatx(1.0),normalizer_neg)
        #分别对存在目标的锚框和背景锚框的分类损失进行归一化
        cls_loss_for_object=keras.backend.sum(cls_loss_for_object)/normalizer_pos
        cls_loss_for_back=ratio*keras.backend.sum(cls_loss_for_back)/normalizer_neg
        #计算总的分类损失，加权了正负样本的损失
        loss=cls_loss_for_object+cls_loss_for_back
        return loss
    return _cls_loss
#计算平滑L1损失
def smooth_L1(sigma=1.0):  #sigma是一个控制平滑程度的超参数
    sigma_squared=sigma**2
    def _smooth_l1(y_true,y_pred):  #y_true真实标签(目标的位置信息+锚框状态的信息)，y_pred模型的预测结果(回归坐标值)
        regression=y_pred  #预测的回归结果
        regression_target=y_true[:,:,:,-1]  #真实的回归坐标
        anchor_state=y_true[:,:,-1] #锚框的状态
        #找到正样本
        indices=tf.where(keras.backend.equal(anchor_state,1)) #找到正样本索引
        regression=tf.gather_nd(regression,indices) #提取正样本的回归预测值
        regression_target=tf.gather_nd(regression_target,indices) #提取正样本的真实目标值
        #计算平滑L1损失
        regression_diff=regression-regression_target
        regression_diff=keras.backend.abs(regression_diff)
        regression_loss=tf.where(keras.backend.less(regression_diff,1.0/sigma_squared),0.5*sigma_squared*keras.backend.pow(regression_diff,2),
                                 regression_diff-0.5/sigma_squared)
        #计算损失的规范化值
        normalizer=keras.backend.maximum(1,keras.backend.shape(indices)[0])
        normalizer=keras.backend.cast(normalizer,dtype=keras.backend.floatx())  #统计正样本的数量
        loss=keras.backend.sum(regression_loss)/normalizer  #损失值总和/正样本数量=最终损失值
        return loss
    return _smooth_l1

#定义回归损失函数，用于计算目标边界框的位置回归损失
def class_loss_regr(num_classes):  #参数num_classes表示目标检测任务中的类别数量
    epsilon=1e-4  #定义小常量epsilon，用于除0错误
    def class_loss_regr_fixed_num(y_true,y_pred):
        x=y_true[:,:,4*num_classes:]-y_pred
        x_abs=K.abs(x)
        x_bool=K.cast(K.less_equal(x_abs,1.0),'float32')
        loss=4*K.sum(y_true[:,:,:4*num_classes]*(x_bool*(0.5*x*x)+(1-x_bool)*(x_abs-0.5)))/K.sum(epsilon+y_true[:,:,:4*num_classes])
        return loss
    return class_loss_regr_fixed_num
#定义分类损失函数，用于计算目标类别的分类损失(交叉熵损失)
def class_loss_cls(y_true,y_pred):
    return K.mean(categorical_crossentropy(y_true[0,:,:],y_pred[0,:,:]))

#图像处理辅助函数
#根据最小边的长度调整图像的尺寸
def get_new_img_size(width,height,img_min_side=600):
    if width<=height:
        f=float(img_min_side)/width
        resized_height=int(f*height)
        resized_width=int(img_min_side)
    else:
        f=float(img_min_side)/height
        resized_width=int(f*width)
        resized_height=int(img_min_side)
    return resized_width,resized_height
#计算CNN网络输出特征图的尺寸
def get_img_output_length(width,height):
    def get_output_length(input_length):
        filter_sizes=[7,3,1,1]
        padding=[3,1,0,0]
        stride=2
        for i in range(4):
            input_length=(input_length+2*padding[i]-filter_sizes[i])//stride+1
        return input_length
    return get_output_length(width),get_output_length(height)

#数据生成类定义
class Generator(object):
    #初始化方法，设置一些基本属性，bbox_util(用于处理目标框)，train_lines(训练数据路径列表)
    def __init__(self,bbox_util,train_lines,num_classes,solid,solid_shape=[600,600]):
        self.bbox_util=bbox_util
        self.train_lines=train_lines
        self.train_batches=len(train_lines)
        self.num_classes=num_classes
        self.solid=solid
        self.solid_shape=solid_shape  
    #这个方法用于实时数据增强和预处理，包括随机调整图像的大小，颜色，翻转等，并调整目标框的坐标
    def get_random_data(self,annotation_line,random=True,jitter=.1,hue=.1,sat=1.1,val=1.1,proc_img=True):
        line=annotation_line.split()  #列表line包含图像文件路径和一系列边界框的信息
        image=Image.open(line[0])
        iw,ih=image.size
        if self.solid:
            w,h=self.solid_shape
        else:
            w,h=get_new_img_size(iw,ih)
        box=np.array([np.array(list(map(int,box.split(','))))for box in line[1:]])  #将Line中的边界框信息提取出来，存在box中
        #以下是图像增强操作
        #图像大小调整
        new_ar=w/h*rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale=rand(.9,1.1)
        if new_ar<1:
            nh=int(scale*h)
            nw=int(nh*new_ar)
        else:
            nw=int(scale*w)
            nh=int(nw/new_ar)
        image=image.resize((nw,nh),Image.BICUBIC)
        #随机平移
        dx=int(rand(0,w-nw))
        dy=int(rand(0,h-nh))
        new_image=Image.new('RGB',(w,h),(128,128,128))
        new_image.paste(image,(dx,dy))
        image=new_image
        #随机水平翻转
        flip=rand()<.5
        if flip:image=image.transpose(Image.FLIP_LEFT_RIGHT)
        #随机色相、饱和度和亮度调整(色相hue，饱和度saturation，明度value)
        hue=rand(-hue,hue)
        sat=rand(1,sat) if rand()<.5 else 1/rand(1,sat)
        val=rand(1,val) if rand()<.5 else 1/rand(1,val)
        x=rgb_to_hsv(np.array(image/255.))
        x[...,0]+=hue
        x[...,0][x[...,0]>1]-=1
        x[...,0][x[...,0]<0]+=1
        x[...,1]*=sat
        x[...,2]*=val
        x[x>1]=1  
        x[x<0]=0   #确保所有通道在0-1之间，超出设为1，小于设为0
        image_data=hsv_to_rgb(x)*255
        #边界框的调整
        #创建一个形状为(len(box),5)的全零数组，用于存储边界框数据
        box_data=np.zeros((len(box),5))
        #如果存在边界框(box数组非空)
        if len(box)>0:
            #随机打乱box数组的顺序
            np.random.shuffle(box)
            #对边界框坐标进行一系列变换
            box[:,[0,2]]=box[:,[0,2]]*nw/iw+dx
            box[:,[1,3]]=box[:,[1,3]]*nh/ih+dy
            #如果需要翻转图像，调整边界框的坐标
            if flip:
                box[:,[0,2]]=w-box[:,[2,0]]
            #确保边界框的坐标在图像范围内
            box[:,0:2][box[:,0:2]<0]=0
            box[:,2][box[:,2]>w]=w
            box[:,3][box[:,3]>h]=h
            #计算边界框的宽度和高度
            box_w=box[:,2]-box[:,0]
            box_h=box[:,3]-box[:,1]
            #保留有效的边界框（宽度和高度都大于1）
            box=box[np.logical_and(box_w>1,box_h>1)]
            #将处理后的边界框数据存储到box_data数组中
            box_data[:len(box)]=box
        #如果没有有效的边界框，返回原始图像数据和一个空的边界框列表
        if len(box)==0:
            return image_data,[]
        #如果存在有效的边界框，返回原始图像数据和边界框数据
        if (box_data[:,:4]>0).any():
            return image_data,box_data
        else:
            return image_data,[]
    #这是一个生成器方法，它使用yield语句生成训练数据的批次，包括图像数据和目标框数据
    def generate(self):
        while True:
            shuffle(self.train_lines) #打乱训练数据的顺序
            lines=self.train_lines
            for annotation_line in lines:
                img,y=self.get_random_data(annotation_line) #遍历每一行注释数据
                height,width,_=np.shape(img)
                if len(y)==0:
                    continue
                # 归一化真实框的坐标
                boxes=np.array(y[:,:4],dtype=np.float32)  #获取真实框的坐标
                boxes[:,0]=boxes[:,0]/width
                boxes[:,1]=boxes[:,1]/height
                boxes[:,2]=boxes[:,2]/width
                boxes[:,3]=boxes[:,3]/height  #归一化真实框的坐标
                #计算真实框的高度和宽度
                box_heights=boxes[:,3]-boxes[:,1]
                box_widths=boxes[:,2]-boxes[:,0]
                #如果真实框的高度或宽度小于0，则跳过该样本
                if (box_heights<=0).any() or (box_widths<=0).any():
                    continue
                y[:,:4]=boxes[:,:4]  #更新真实框的坐标
                #获取先验框
                anchors=get_anchors(get_img_output_length(width,height),width,height)
                # 计算真实框对应的先验框，与这个先验框应有的预测结果
                assignment=self.bbox_util.assign_boxes(y,anchors)
                num_regions=256
                classification=assignment[:,4]
                regression=assignment[:,:]
                # 随机选择一部分正样本和负样本
                mask_pos=classification[:]>0  #创建布尔掩码，标识哪些是正样本
                num_pos=len(classification[mask_pos])  #计算正样本数量
                if num_pos >num_regions/2:
                    val_locs=random.sample(range(num_pos),int(num_pos-num_regions/2))  #随机抽取
                    classification[mask_pos][val_locs]=-1  #设置被抽样正样本的分类标签为-1，示不会参与训练
                    regression[mask_pos][val_locs,-1]=-1   #设置被抽样正样本的回归信息最后一列为-1，表示不会参与训练

                mask_neg=classification[:]==0  #标识哪些是负样本
                num_neg=len(classification[mask_neg]) #计算负样本数量
                if len(classification[mask_neg])+num_pos>num_regions:
                    val_locs=random.sample(range(num_neg),int(num_neg-num_pos))
                    classification[mask_neg][val_locs]=-1  #剔除部分负样本
                
                classification=np.reshape(classification,[-1,1])
                regression=np.reshape(regression,[-1,5])  #使它们从原来的形状变成了二维数组

                tmp_inp=np.array(img) #存储了图像数据的副本
                #两个元素的列表：经过处理的分类和回归数据，它们被扩展为适应模型输入的形状
                tmp_targets=[np.expand_dims(np.array(classification,dtype=np.float32),0),np.expand_dims(np.array(regression,dtype=np.float32),0)]
                #yield语句：生成器，每次迭代产生三个值，预处理后的图像输入，处理后的目标数据和原始的y数据
                yield preprocess_input(np.expand_dims(tmp_inp,0)),tmp_targets,np.expand_dims(y,0)




