import numpy as np
from config.Config import Config

config = Config()


# 生成base_anchor
def generate_anchor_base(base_size=16, ratios=None, anchor_scales=None):
    if not ratios:
        ratios = config.ratios
    if not anchor_scales:
        anchor_scales = config.anchor_scales

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base


'''
通过width：(0-60)*16，height(0-40)*16建立shift偏移量数组，
再和base_anchor基准坐标数组累加，
得到特征图上所有像素对应的Anchors的坐标值，是一个[21600,4]的数组。

'''


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # 计算网格中心点
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(),
                      shift_x.ravel(), shift_y.ravel(),), axis=1)

    # 每个网格点上的9个先验框
    A = anchor_base.shape[0]  # 9 每个网格点上的9个先验框
    K = shift.shape[0]  # width/feat_stride * height/feat_stride  有多少个像素点
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
    # 所有的先验框
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


