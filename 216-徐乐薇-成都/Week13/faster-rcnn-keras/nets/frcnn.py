from nets.resnet import ResNet50,classifier_layers                                                                                                          # 导入ResNet50,classifier_layers
from keras.layers import Conv2D,Input,TimeDistributed,Flatten,Dense,Reshape                                                                                 # 导入Conv2D,Input,TimeDistributed,Flatten,Dense,Reshape
from keras.models import Model                                                                                                                              # 导入Model,用于构建模型
from nets.RoiPoolingConv import RoiPoolingConv                                                                                                              # 导入RoiPoolingConv,用于ROI池化层
# 该文件定义了faster-rcnn的网络结构，包括rpn网络和分类网络

# 该函数用于构建rpn网络
def get_rpn(base_layers, num_anchors):                                                                                                                      # rpn网络，用于生成建议框，输入为resnet50的输出，输出为建议框的分类和回归
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)                                  # 3x3卷积，512个通道，padding为same，激活函数为relu

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)                                      # 1x1卷积，输出通道数为num_anchors，激活函数为sigmoid
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)                                     # 1x1卷积，输出通道数为num_anchors*4，激活函数为linear。每个建议框有4个坐标，分别为x1,y1,x2,y2，所以输出通道数为num_anchors*4
    
    x_class = Reshape((-1,1),name="classification")(x_class)                                                                                                # 将x_class的形状变为(-1,1)，其中-1表示不确定，1表示1个通道
    x_regr = Reshape((-1,4),name="regression")(x_regr)                                                                                                      # 将x_regr的形状变为(-1,4)，其中-1表示不确定，4表示4个通道
    return [x_class, x_regr, base_layers]                                                                                                                   # 返回x_class,x_regr,base_layers

# 该函数用于构建分类网络
def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):                                                                      # 分类网络，用于对建议框进行分类，输入为resnet50的输出和建议框，输出为建议框的分类和回归
    pooling_regions = 14                                                                                                                                    # 池化区域的大小
    input_shape = (num_rois, 14, 14, 1024)                                                                                                                  # 输入的形状
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])                                                                     # RoiPoolingConv层，调用接口，输入为resnet50的输出和建议框，输出为建议框的池化特征
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)                                                                          # classifier_layers层，输入为建议框的池化特征，输出为建议框的分类和回归
    out = TimeDistributed(Flatten())(out)                                                                                                                   # 将out的形状变为(-1,1)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)          # TimeDistributed层，输入为out，输出为建议框的分类
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)  # TimeDistributed层，输入为out，输出为建议框的回归
    return [out_class, out_regr]                                                                                                                            # 返回out_class,out_regr

# 该函数用于构建模型
def get_model(config,num_classes):                                                                                                                          # 构建模型，输入为config和num_classes，输出为model_rpn,model_classifier,model_all
    inputs = Input(shape=(None, None, 3))                                                                                                                   # inputs的形状为(None,None,3),即图片的大小
    roi_input = Input(shape=(None, 4))                                                                                                                      # roi_input的形状为(None,4),即建议框的坐标,4为(xmin,ymin,xmax,ymax)
    base_layers = ResNet50(inputs)                                                                                                                          # 调用ResNet50，输入为inputs，输出为base_layers

    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)                                                                             # 计算建议框的数量
    rpn = get_rpn(base_layers, num_anchors)                                                                                                                 # 调用get_rpn，输入为base_layers和num_anchors，输出为rpn
    model_rpn = Model(inputs, rpn[:2])                                                                                                                      # 构建model_rpn，输入为inputs，输出为rpn[:2]

    classifier = get_classifier(base_layers, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)                                            # 调用get_classifier，输入为base_layers,roi_input,config.num_rois,nb_classes,trainable，输出为classifier
    model_classifier = Model([inputs, roi_input], classifier)                                                                                               # 构建model_classifier，输入为inputs,roi_input，输出为classifier

    model_all = Model([inputs, roi_input], rpn[:2]+classifier)                                                                                              # 构建model_all，输入为inputs,roi_input，输出为rpn[:2]+classifier
    return model_rpn,model_classifier,model_all

# 该函数用于构建预测模型
def get_predict_model(config,num_classes):                                                                                                                  # 预测模型，输入为config和num_classes，输出为model_rpn,model_classifier_only
    inputs = Input(shape=(None, None, 3))                                                                                                                   # inputs的形状为(None,None,3),即图片的大小
    roi_input = Input(shape=(None, 4))                                                                                                                      # roi_input的形状为(None,4),即建议框的坐标,4为(xmin,ymin,xmax,ymax)
    feature_map_input = Input(shape=(None,None,1024))                                                                                                       # feature_map_input的形状为(None,None,1024),即建议框的池化特征

    base_layers = ResNet50(inputs)                                                                                                                          # 调用ResNet50，输入为inputs，输出为base_layers
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)                                                                             # 计算建议框的数量，即num_anchors，配置的是3*3=9，共9种建议框
    rpn = get_rpn(base_layers, num_anchors)                                                                                                                 # 调用get_rpn，输入为base_layers和num_anchors，输出为rpn，rpn的形状为(None,None,9),(None,None,9*4)，4为(x,y,w,h)
    model_rpn = Model(inputs, rpn)                                                                                                                          # 构建model_rpn，输入为inputs，输出为rpn，即建议框的分类和回归，分类为9种，回归为36种，9种建议框*4个坐标，4个坐标分别为x,y,w,h

    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)                                      # 调用get_classifier，输入为feature_map_input,roi_input,config.num_rois,nb_classes,trainable，输出为classifier
    model_classifier_only = Model([feature_map_input, roi_input], classifier)                                                                               # 构建model_classifier_only，输入为feature_map_input,roi_input，输出为classifier

    return model_rpn,model_classifier_only                                                                                                                  # 返回model_rpn,model_classifier_only