#定义了一些函数，用于构建目标检测模型，包括RPN模型和分类器模型，并提供了创建完整模型和预测模型的功能


from nets.resnet import ResNet50,classifier_layers
from keras.layers import Conv2D,Input,TimeDistributed,Flatten,Dense,Reshape
from keras.models import Model
from nets.RoiPoolingConv import RoiPoolingConv
#用于构建Region Proposal Network (RPN)，通过一系列的卷积操作，在输入特征图上生成了两个输出，分类概率x_class和边界框坐标回归值x_regr
def get_rpn(base_layers, num_anchors):  #base_layers卷积网络的输出特征图，num_anchors每个锚框的数量
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    
    x_class = Reshape((-1,1),name="classification")(x_class)
    x_regr = Reshape((-1,4),name="regression")(x_regr)
    return [x_class, x_regr, base_layers]
#用于构建目标分类网络，通过RoiPoolingConv将每个ROI区域从输入特征图中提取出来，并通过一系列全连接层进行分类和边界框回归预测
#base_layers输出特征图，input_rois输入的ROI，num_roisROI的数量，nb_classes类别数，trainable是否可训练
def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False): 
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]

#构建完整的目标检测模型，它调用了get_rpn和get_classifier函数，构建了RPN模型和分类器模型，返回三个模型，RPN模型model_rpn，分类器模型model_classifier，整体模型model_all
#整体模型可以接受图像和ROI输入，并输出RPN的分类概率，边界框回归值以及分类器的分类概率和边界框回归值。
def get_model(config,num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    base_layers = ResNet50(inputs)

    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn[:2])

    classifier = get_classifier(base_layers, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier = Model([inputs, roi_input], classifier)

    model_all = Model([inputs, roi_input], rpn[:2]+classifier)
    return model_rpn,model_classifier,model_all

# 用于构建预测模型，它调用了get_rpn和get_classifier函数，仅返回了RPN模型和仅包括分类器部分的预测模型。
# 预测模型可以接受特征图、ROI输入，并输出分类器的分类概率和边界框回归值
def get_predict_model(config,num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None,None,1024))

    base_layers = ResNet50(inputs)
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn)

    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    return model_rpn,model_classifier_only