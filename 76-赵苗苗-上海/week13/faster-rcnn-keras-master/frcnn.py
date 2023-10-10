"""这段代码是使用Faster R-CNN模型进行目标检测,根据预训练权重和分类信息，加载模型并对图像进行检测，最后返回带有检测结果的图像"""
import cv2
import keras
import numpy as np
import colorsys
import pickle
import os
import nets.frcnn as frcnn
from nets.frcnn_training import get_new_img_size
from keras import backend as K
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image,ImageFont, ImageDraw
from utils.utils import BBoxUtility
from utils.anchors import get_anchors
from utils.config import Config
import copy
import math

#定义名为FRCNN的类,包含一个类属性_defaults，它是一个字典，包含一些默认参数值，model_path模型路径，classes_path类别路径，confidence置信度阈值
class FRCNN(object):
    _defaults={
        'model_path':r"D:\AI\submit assignment\Assignment\week13\faster-rcnn-keras-master\model_data\voc_weights.h5",
        'classes_path':r"D:\AI\submit assignment\Assignment\week13\faster-rcnn-keras-master\model_data\voc_classes.txt",
        'confidence':0.7
    }
    #定义静态方法get_defaults，用@classmethod​装饰器修饰，cls类对象本身，n传入的参数，检查参数n是否在_defaults字典的键中
    @classmethod
    def get_default(cls,n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name'" + n +"'"
    #类的初始化部分，初始化Faster R-CNN模型实例，加载模型权重和类别信息
    def __init__(self,**kwargs):
        #将默认参数更新到实例对象的属性中
        self.__dict__.update(self._defaults)
        #获取类别信息
        self.class_names=self._get_class()
        #获取当前会话
        self.sess=K.get_session()
        #创建配置对象
        self.config=Config()
        #生成模型
        self.generate()
        #创建BBoxUtility对象
        self.bbox_util=BBoxUtility()
    
    #获取类别信息：它读取指定路径的文件，提取文件中的类别信息，并以列表形式返回这些类别名称
    def _get_class(self):   #self表示实例对象本身
        classes_path=os.path.expanduser(self.classes_path)  #将路径中的波浪号~扩展为用户主目录
        with open(classes_path) as f:
            class_names=f.readlines()
        class_names=[c.strip() for c in class_names]  #对class_names列表处理，使用strip()方法去除字符串开头和结尾的空白符，生成新的列表
        return class_names
    #生成模型：根据配置和类别信息，生成Faster R-CNN模型，加载预训练权重，并设置颜色信息
    def generate(self):
        #扩展和规范化模型路径model_path，使用断言确保模型文件的扩展名是.h5
        model_path=os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'),'Keras model or weights must be a .h5 file.'
        
        #计算总类别数，确定需要检测的类别总数，包括物体类别和背景类别(+1就是背景类别)
        self.num_classes=len(self.class_names)+1
        
        #根据配置和类别数构建Faster R-CNN模型中的RPN和分类器模型，分别赋值给self.model_rpn，self.model_classifier
        #在整个Faster R-CNN模型中，RPN模型负责生成候选区域，这些区域将被送入后续的RoI Pooling层，然后通过分类器进行目标检测和物体分类
        #RPN模型用于在输入图像上滑动一个固定尺寸的窗口，并为每个窗口提出两方面的建议，计算出窗口内是否有物体，还回归边界框以预测物体位置
        #RPN模型并不是完全密集地滑动窗口，而是在图像的特征图上进行滑动，在特征图上的每个位置进行锚框的生成和预测
        #RPN模型在输入图像上滑动的窗口数量取决于特征图的尺寸，以及锚框的尺度和长宽比的组合数量
        #get_predict_model是自定义函数，用于生成 Faster R-CNN 模型
        self.model_rpn,self.model_classifier=frcnn.get_predict_model(self.config,self.num_classes)
        
        #分别使用load_weights()方法加载预训练权重到RPN模型和分类器模型中
        self.model_rpn.load_weights(self.model_path,by_name=True)
        self.model_classifier.load_weights(self.model_path,by_name=True,skip_mismatch=True)
        #打印出信息表示模型路径model_path已经加载完毕，并且与之关联的锚框和分类数量也提前加载好了
        print('{} model,anchors,and classes loaded.'.format(model_path))

        #通过对类别数量进行归一化，计算不同类别框的颜色，并将颜色信息存储在self.colors列表中，用于后续对象检测结果的可视化
        #通过列表推导式创建一个由元组组成的列表，每个元组包含三个值，H(色调)，S(饱和度)，V(亮度)，H取值0-1
        hsv_tuples=[(x/len(self.class_names),1.,1.) for x in range(len(self.class_names))]
        #使用map函数和lambda表达式将HSV格式的元组转换为RGB格式的元组，hsv_to_rgb函数将HSV值转换为RGB值
        self.colors=list(map(lambda x:colorsys.hsv_to_rgb(*x),hsv_tuples))
        #使用map函数和lambda表达式将RGB值的元组中的浮点数值（范围在0-1）转换为整型（范围在0-255），以便在可视化时使用标准RGB值
        self.colors=list(map(lambda x:(int(x[0]*255),int(x[1]*255),int(x[2]*255)),self.colors))

     #辅助函数部分：根据输入图像的尺寸计算输出特征图的尺寸
    def get_img_output_length(self,width,height):
        #计算输出长度
        def get_output_length(input_length):
            filter_sizes=[7,3,1,1]  #卷积核大小
            padding=[3,1,0,0]       #填充的数量，这里表示，第一层上下左右各填充3个元素，第二层一个元素，三四层不进行填充
            stride=2
            #遍历了四个层次的卷积操作(或池化操作)
            for i in range(4):
                 #考虑了输入长度，填充数量，卷积核大小和步幅的影响
                input_length=(input_length+2*padding[i]-filter_sizes[i])//stride+1  
            return input_length
        return get_output_length(width),get_output_length(height)

    #图像检测部分：使用预训练的模型进行目标检测，识别输入图像中的目标并在图像上绘制出检测框和对应的类别标签
    def detect_image(self,image):
        #获取输入图像的形状信息，并初始化变量
        image_shape=np.array(np.shape(image)[0:2])
        old_width=image_shape[1]
        old_height=image_shape[0]
        old_image=copy.deepcopy(image)
        width,height=get_new_img_size(old_width,old_height)  #调用get_new_img_size函数，返回调整后的图像宽度和高度

        #对图像进行预处理，包括调整图像的大小，将图像转换为numpy数组，归一化处理，并使用预训练模型进行目标检测的预测，得到预测结果preds
        image=image.resize([width,height])  #调整图像大小宽高为width,height
        photo=np.array(image,dtype=np.float64)
        photo=preprocess_input(np.expand_dims(photo,0)) #对图像进行归一化预处理，将图像从原始形状(height, width, channels)​ 扩展为 ​(1, height, width, channels)​
        preds=self.model_rpn.predict(photo)  #使用RPN模型对预处理后的图像photo进行预测，得到预测结果preds(每一行对应一个目标物体的预测结果，每列代表目标物体的边界框位置，分类概率，分类标签等信息)
        
        #生成与输入图像大小相匹配的锚框并使用预测结果preds和锚框进行目标框的解码，得到目标框R
        anchors=get_anchors(self.get_img_output_length(width,height),width,height) #调用get_anchors函数生成锚框（用于生成的边界框与真实目标的匹配）
        rpn_results=self.bbox_util.detection_out(preds,anchors,1,confidence_threshold=0) #rpn_results每一行对应一个目标物体的预测结果，每列包含了目标物体的边界框位置(x,y,高度,宽度)，目标得分，类别标签等信息
        R=rpn_results[0][:,2:]  #R包含了目标框的位置和大小信息，每行对应一个目标框，每列代表目标框的不同属性(x,y,宽度,高度)
        #调整目标狂的坐标和大小，并获取预测结果preds中的中间特征图base_layer
        R[:,0]=np.array(np.round(R[:,0]*width/self.config.rpn_stride),dtype=np.int32) 
        R[:,1]=np.array(np.round(R[:,1]*height/self.config.rpn_stride),dtype=np.int32)
        R[:,2]=np.array(np.round(R[:,2]*width/self.config.rpn_stride),dtype=np.int32)
        R[:,3]=np.array(np.round(R[:,3]*height/self.config.rpn_stride),dtype=np.int32) #1,2,3,4列分别指目标框的左上角x坐标，左上角y坐标，宽度，高度
        R[:,2]-=R[:,0]  #计算目标框的高度
        R[:,3]-=R[:,1]  #计算目标框的宽度
        base_layer=preds[2]  #base_layer是特征图

        #对筛选掉宽度(r[2])或高度(r[3])小于1的目标框，生成新的目标框数组R。然后遍历每个目标框，并使用分类器模型进行分类和回归预测，得到类别概率P_cls和回归参数P_regr
        delete_line=[]
        for i,r in enumerate(R):
            if r[2]<1 or r[3]<1:
                delete_line.append(i)
        R=np.delete(R,delete_line,axis=0)  #筛选掉宽度(r[2])或高度(r[3])小于1的目标框，生成新的目标框数组R

        bboxes=[]
        probs=[]
        labels=[]  #创建三个空列表，存储目标框的坐标、置信度、类别标签
        #对目标框进行划分和处理，以便后续使用模型进行预测
        for jk in range(R.shape[0]//self.config.num_rois+1): #R.shape[0]指目标框的数量，self.config.num_rois指每个训练批次中要处理的目标框(ROI)的数量
            ROIs=np.expand_dims(R[self.config.num_rois*jk:self.config.num_rois*(jk+1),:],axis=0)#ROIs=(1,num_rois, num_columns),num_rois所选择的目标框的数量, num_columns每个目标框的属性或特征的数量
            if ROIs.shape[1]==0:
                break
            if jk==R.shape[0]//self.config.num_rois: #右侧的实际意义：确定需要多少个完整的批次训练来处理所有目标框
                curr_shape=ROIs.shape  #将ROIs的形状赋值给变量curr_shape,(1, self.config.num_rois, num_columns)​，(批次大小，目标框的数量，每个目标框的属性或特征的数量)
                target_shape=(curr_shape[0],self.config.num_rois,curr_shape[2])  #定义目标形状
                ROIs_padded=np.zeros(target_shape).astype(ROIs.dtype) #创建全零数组，形状与target_shape相同，数据类型与ROIs相同
                ROIs_padded[:,:curr_shape[1],:]=ROIs  #将ROIs的目标框坐标信息复制到ROIs_padded中，以匹配目标形状的大小
                ROIs_padded[0,curr_shape[1]:,:]=ROIs[0,0,:]  #将ROIs的第一个目标框的坐标信息复制到ROIs_padded中，以填充超出目标形状范围的部分
                ROIs=ROIs_padded  #更新ROIs变量的值
            [P_cls,P_regr]=self.model_classifier.predict([base_layer,ROIs])

        #根据类别概率的最大值和阈值筛选目标框
            for ii in range(P_cls.shape[1]):  #P_cls(1, num_rois, num_classes)（1，目标框数量，类别数量）；P_regr (1, num_rois, 4 * num_classes)
                if np.max(P_cls[0,ii,:])<self.confidence or np.argmax(P_cls[0,ii,:])==(P_cls.shape[2]-1):
                  continue
                label=np.argmax(P_cls[0,ii,:]) #获取目标框ii的类别标签，即类别预测概率最高的索引
                (x,y,w,h)=ROIs[0,ii,:]  #获取目标框ii的坐标和尺寸信息
                cls_num=np.argmax(P_cls[0,ii,:]) #获取目标框ii的类别标签索引
                (tx,ty,tw,th)=P_regr[0,ii,4*cls_num:4*(cls_num+1)]  #tx,ty,tw,th分别表示目标框x,y,w,h四个方面的回归值
                tx/=self.config.classifier_regr_std[0]
                ty/=self.config.classifier_regr_std[1]
                tw/=self.config.classifier_regr_std[2]
                th/=self.config.classifier_regr_std[3]  #将回归值除以预定义的分类器回归标准化因子，以将回归值变换为相对于基准尺度的调整量
                #回归系数tx,ty,tw,th分别用于调整目标框的中心点坐标、宽度、高度，并计算出调整后的目标框的左上角和有下角的坐标，转换为整数后，得到最终的整数轴对齐的目标框范围
                cx=x+w/2 #计算目标框的中心点的x坐标
                cy=y+h/2 #计算目标框的中心点的y坐标
                cx1=tx*w+cx #调整目标框中心点的x坐标
                cy1=ty*h+cy #调整目标框中心点的y坐标
                w1=math.exp(tw)*w #调整后的目标框宽度
                h1=math.exp(th)*h #调整后的目标框高度
                x1=cx1-w1/2 #计算目标框的左上角的x坐标
                y1=cy1-h1/2 #计算目标框的左上角的y坐标
                x2=cx1+w1/2 #计算目标框的右上角的x坐标
                y2=cy1+h1/2 #计算目标框的右上角的y坐标
                x1=int(round(x1))
                y1=int(round(y1))
                x2=int(round(x2))
                y2=int(round(y2))  #取整
                #将满足条件的目标框的坐标，类别概率和标签加到对应的数组中，并在没有检测到目标框时，返回原始图像
                bboxes.append([x1,y1,x2,y2])
                probs.append(np.max(P_cls[0,ii,:]))
                labels.append(label)   #存储目标框的坐标、置信度、类别标签
        if len(bboxes)==0:
            return old_image

        #对目标框进行非极大值抑制(NMS)操作，去除重叠度高的目标框，并将目标框的坐标还原到原始图像的尺度上
        labels=np.array(labels)
        probs=np.array(probs)
        boxes=np.array(bboxes,dtype=np.float32)
        #将boxes的左上角和右下角的坐标从图像坐标系转换为特征图坐标系
        boxes[:,0]=boxes[:,0]*self.config.rpn_stride/width
        boxes[:,1]=boxes[:,1]*self.config.rpn_stride/height
        boxes[:,2]=boxes[:,2]*self.config.rpn_stride/width
        boxes[:,3]=boxes[:,3]*self.config.rpn_stride/height
        #调用 ​self.bbox_util​ 对象的 ​nms_for_out()​ 函数，进行非极大值抑制(NMS)处理，来得到预测结果中最佳候选框列表results
        results=np.array(self.bbox_util.nms_for_out(np.array(labels),np.array(probs),np.array(boxes),self.num_classes-1,0.4))
        #从results中提取出最佳候选框的标签索引、置信度和坐标信息
        top_label_indices=results[:,0]
        top_conf=results[:,1]
        boxes=results[:,2:]
        #将boxes的坐标信息从特征图坐标系重新映射回原始图像坐标系，使用缩放因子将其调整至原始图像中的位置
        boxes[:,0]=boxes[:,0]*old_width
        boxes[:,1]=boxes[:,1]*old_height
        boxes[:,2]=boxes[:,2]*old_width
        boxes[:,3]=boxes[:,3]*old_height
        #创建一个指定字体文件和大小的ImageFont对象，用于在图像上绘制文本时使用的字体
        #ImageFont.truetype创建一个字体对象，'model_data/simhei.ttf'​指定字体文件的路径和文件名，size用于指定字体大小的参数
        font=ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2*np.shape(image)[1]+0.5).astype('int32'))
        #计算图像绘制边界框时所需的线条粗细，然后将原始图像赋值给变量image
        thickness=(np.shape(old_image)[0]+np.shape(old_image)[1])//width
        image=old_image
        #遍历保留下来的目标框，在图像上绘制框和类别标签，最后返回绘制了检测框和类别标签的图像
        for i,c in enumerate(top_label_indices):
            #获取预测类别名称和置信度分数
            predicted_class=self.class_names[int(c)]  #根据c的值获取对应的预测类别名称
            score=top_conf[i]  #获取对应目标的置信度分数
            #调整目标框边界大小
            left,top,right,bottom=boxes[i] #将目标框的左上角和右下角坐标分别赋值给变量
            top=top-5
            left=left-5
            bottom=bottom+5
            right=right+5  #扩大目标框的尺寸
            #四舍五入并取整(确保目标框的上、左、下、右边界在不超过图像边界的前提下，进行调整)
            top=max(0,np.floor(top+0.5).astype('int32'))
            left=max(0,np.floor(left+0.5).astype('int32')) #确保top，left不小于0
            bottom=min(np.shape(image)[0],np.floor(bottom+0.5).astype('int32'))
            right=min(np.shape(image)[1],np.floor(right+0.5).astype('int32')) #比较舍入后的值与图像的高度和宽度，保证不超过图像的边界
            #在图像上绘制检测结果的标签和边框
            label='{} {:.2f}'.format(predicted_class,score)
            draw=ImageDraw.Draw(image)  #创建一个绘制对象
            label_size=draw.textsize(label,font)  #计算标签的大小
            label=label.encode('utf-8') 
            print(label)
            #根据标签文本的高度和边框的位置，确定文本的起始位置
            if top-label_size[1]>=0:
                text_origin=np.array([left,top-label_size[1]])
            else:
                text_origin=np.array([left,top+1])
            #使用for循环绘制边框
            for i in range(thickness):  #thickness指绘制边框时线条的宽度
                draw.rectangle(
                    [left+i,top+i,right-i,bottom-i],outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin),tuple(text_origin+label_size)],fill=self.colors[int(c)])
            draw.text(text_origin,str(label,'UTF-8'),fill=(0,0,0),font=font)  #使用draw.tex函数在图像上绘制文本
            del draw  #删除绘制对象draw
        return image
    #会话关闭部分：关闭会话，释放资源
    def close_session(self):
        self.sess.close()






    

