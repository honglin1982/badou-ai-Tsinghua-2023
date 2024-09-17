from keras.layers import Input, Conv2D, Reshape, TimeDistributed, Flatten, Dense
from keras import Model
from .resnet import ResNet50, classifier_layers
from .RoiPoolingConv import RoiPoolingConv


def get_rpn(base_layers, num_anchors):
    """
        创建rpn网络
        Parameters
        ----------
        base_layers：resnet50输出的特征层（None,38,38,1024）
        num_anchors：先验框框数量，通常为9，即每个网格分配有9个先验框

        Returns
        -------
    """
    # 38x38x1024 -> 38x38x512
    x = Conv2D(512, (3, 3), padding="same", activation='relu', kernel_initializer='normal',
               name='rpn_conv1')(base_layers)
    # 38x38x512 -> 38x38x9
    # 利用一个1x1卷积调整通道数，获得预测结果
    # rpn_class只预测该先验框是否包含物体
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                     name='rpn_out_class')(x)
    # 38x38x512 -> 38x38x36
    # 预测每个先验框的变化量，4代表变化量的x,y,w,h
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                    name='rpn_out_regress')(x)

    x_class = Reshape((-1, 1), name='classification')(x_class)
    x_regr = Reshape((-1, 4), name='regression')(x_regr)

    return [x_class, x_regr, base_layers]


def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    # 使用卷积代替全连接
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    out = TimeDistributed(Flatten())(out)
    # 分别的全连接
    out_class = TimeDistributed(
        Dense(
            nb_classes, activation='softmax', kernel_initializer='zero'
        ),
        name=f'dense_class_{nb_classes}'
    )(out)
    out_regr = TimeDistributed(
        Dense(
            4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'
        ),
        name=f'dense_regress_{nb_classes}'
    )(out)

    return [out_class, out_regr]


def get_model(config, num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    base_layers = ResNet50(inputs)

    # 每个网格的先验框个数（枚举）  （3x3）
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn[:2])

    # 后面的分类网络
    classifier = get_classifier(base_layers, roi_input, config.num_rois, nb_classes=num_classes,
                                trainable=True)
    model_classifier = Model([inputs, roi_input], classifier)

    model_all = Model([inputs, roi_input], rpn[:2] + classifier)

    return model_rpn, model_classifier, model_all


def get_predict_model(config, num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None, None, 1024))

    base_layers = ResNet50(inputs)
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn)

    classifier = get_classifier(feature_map_input, roi_input, config.num_rois,
                                nb_classes=num_classes, trainable=True)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    return model_rpn, model_classifier_only

