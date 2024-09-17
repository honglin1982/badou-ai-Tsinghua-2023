from nets.resnet import ResNet50,classifier_layers
from keras.layers import Conv2D,Input,TimeDistributed,Flatten,Dense,Reshape
from keras.models import Model
from nets.RoiPoolingConv import RoiPoolingConv

# 构建RPN
def get_rpn(base_layers, num_anchors):
    # 3x3卷积完成特征图预处理
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    # x_class分类头 判断正负样本，使用1x1卷积实现， x_regr回归头 proposal进行精炼 使用1x1卷积实现
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    # reshape
    x_class = Reshape((-1,1),name="classification")(x_class)
    x_regr = Reshape((-1,4),name="regression")(x_regr)
    return [x_class, x_regr, base_layers]

def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    # RPN输出进行 roi pool 统一rpn输出的尺寸
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    # 分类预处理
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    out = TimeDistributed(Flatten())(out)
    # out_class分类结果  out_regr回归结果
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]

def get_model(config,num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    base_layers = ResNet50(inputs)

    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    # 得到RPN网络 以及 RPN输出
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn[:2])
    # 获取分类网络结果
    classifier = get_classifier(base_layers, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier = Model([inputs, roi_input], classifier)

    model_all = Model([inputs, roi_input], rpn[:2]+classifier)
    return model_rpn,model_classifier,model_all

def get_predict_model(config,num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None,None,1024))
    # 特征提取
    base_layers = ResNet50(inputs)
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    # RPN网络
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn)
    # 分类网络
    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    return model_rpn,model_classifier_only