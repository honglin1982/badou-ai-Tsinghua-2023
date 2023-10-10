#定义了一些与边界框(Bounding Box)相关的函数，主要用于计算两个边界框之间的交并比(IoU)以及处理边界框
import numpy as np
import copy
#计算两个边界框的并集面积
def union(au,bu,area_intersection):
    area_a=(au[2]-au[0])*(au[3]-au[1])
    area_b=(bu[2]-bu[0])*(bu[3]-bu[1])
    area_union=area_a+area_b-area_intersection
    return area_union
#计算两个边界框的交集面积
def intersection(ai,bi):
    x=max(ai[0],bi[0])
    y=max(ai[1],bi[1])
    w=min(ai[2],bi[2])-x
    h=min(ai[3],bi[3])-y
    if w<0 or h<0:
        return 0
    return w*h
#计算两个边界值的IoU值(交/并)
def iou(a,b):
    if a[0]>=a[2] or a[1]>=a[3] or b[0]>=b[2]or b[1]>=b[3]:
        return 0.0
    area_i=intersection(a,b)
    area_u=union(a,b,area_i)
    return float(area_i)/float(area_u+ 1e-6)
#为每一个RoI分配一个类别标签和一个回归目标
def calc_iou(R,config,all_boxes,width,height,num_classes):
    #将所有边界框的坐标转换为相对于特征图的坐标
    bboxes=all_boxes[:,:4]
    #初始化一个与bboxes同样大小的零矩阵，用于存储处理后的真实边界框坐标
    gta=np.zeros((len(bboxes),4))
    #循环部分：将边界框坐标转化为相对于特定步幅的坐标，存储在gta中
    for bbox_num,bbox in enumerate(bboxes):
        gta[bbox_num,0]=int(round(bbox[0]*width/config.rpn_strides))
        gta[bbox_num,1]=int(round(bbox[1]*height/config.rpn_strides))
        gta[bbox_num,2]=int(round(bbox[2]*width/config.rpn_strides))
        gta[bbox_num,3]=int(round(bbox[3]*height/config.rpn_strides))
    #初始化几个空列表，用于存储RoI(感兴趣区域)，类别编号，类别回归坐标，类别回归标签，IoUs
    x_roi=[]
    y_class_num=[]
    y_class_regr_coords=[]
    y_class_regr_label=[]
    IoUs=[]
   
    #遍历R中的每个候选区域
    for ix in range(R.shape[0]): #R.shape[0]表示目标检测模型在处理当前图像时要考虑的潜在目标数量
        x1=R[ix,0]*width/config.rpn_stride
        y1=R[ix,1]*height/config.rpn_stride
        x2=R[ix,2]*width/config.rpn_stride
        y2=R[ix,3]*height/config.rpn_stride

        x1=int(round(x1))
        y1=int(round(y1))
        x2=int(round(x2))
        y2=int(round(y2))
        #初始化变量
        best_iou=0.0   #跟踪最佳IOU值
        best_bbox=-1   #跟踪相应的边界框信息
        
        #计算当前候选框与真实边界框的IoU，并找到最佳匹配的真实边界框
        for bbox_num in range(len(bboxes)):
            curr_iou=iou([gta[bbox_num,0],gta[bbox_num,1],gta[bbox_num,2],gta[bbox_num,3]],[x1,y1,x2,y2])
            if curr_iou>best_iou:
                best_iou=curr_iou
                best_bbox=bbox_num
        
        #如果最佳IoU小于阈值，则跳过当前候选框
        if best_iou<config.classifier_min_overlap:
            continue
        else:
            w=x2-x1
            h=y2-y1
            x_roi.append([x1,y1,w,h])
            IoUs.append(best_iou)
        
        #根据IoU的范围确定标签
        if config.classifier_min_overlap<=best_iou<config.classifier_max_overlap:
            label=-1    #表示当前候选框不是正样本也不是负样本，设置为-1
        elif config.classifier_max_overlap<=best_iou:
            label=int(all_boxes[best_bbox,-1])
            cxg=(gta[best_bbox,0]+gta[best_bbox,2])/2.0
            cyg=(gta[best_bbox,1]+gta[best_bbox,3])/2.0

            cx=x1+w/2.0
            cy=y1+h/2.0

            tx=(cxg-cx)/float(w)
            ty=(cyg-cy)/float(h)
            tw=np.log((gta[best_bbox,2]-gta[best_bbox,0])/float(w))
            th=np.log((gta[best_bbox,3]-gta[best_bbox,1])/float(h))
        else:
            print('roi={}'.format(best_iou))
            raise RuntimeError
        
        #生成分类标签和回归标签
        class_label=num_classes*[0]
        class_label[label]=1
        y_class_num.append(copy.deepcopy(class_label))  #y_class_num 存储了独热编码后的类别信息
        coords=[0]*4*(num_classes-1)  #列表存储目标框的坐标信息
        labels=[0]*4*(num_classes-1)  #列表存储目标框的标签信息
        if label !=-1:
            label_pos=4*label
            sx,sy,sw,sh=config.classifier_regr_std  #classifier_regr_std回归损失计算的标准差
            coords[label_pos:4+label_pos]=[sx*tx,sy*ty,sw*tw,sh*th]
            labels[label_pos:4+label_pos]=[1,1,1,1]
            y_class_regr_coords.append(copy.deepcopy(coords))  #y_class_regr_coords 存储了归一化后的坐标信息
            y_class_regr_label.append(copy.deepcopy(labels))   #y_class_regr_label 存储了目标框的标签信息
        else:
            y_class_regr_coords.append(copy.deepcopy(coords)) 
            y_class_regr_label.append(copy.deepcopy(labels))
    
    #如果没有生成任何候选框，则返回为空
    if len(x_roi)==0:
        return None,None,None,None
    
    #将生成的候选框、分类标签和回归坐标转换为数组，并进行扩展维度
    X=np.array(x_roi)
    Y1=np.array(y_class_num)
    Y2=np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)
    return np.expand_dims(X,axis=0),np.expand_dims(Y1,axis=0),np.expand_dims(Y2,axis=0),IoUs