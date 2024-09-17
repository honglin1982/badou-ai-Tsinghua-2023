import numpy as np
import keras
from .config import Config

config = Config()


def generate_anchors(sizes=None, ratios=None):
    """生成一组原始锚框xminyminxmaxymax"""
    if sizes is None:
        sizes = config.anchor_box_scales

    if ratios is None:
        ratios = config.anchor_box_ratios

    num_anchors = len(sizes) * len(ratios)
    anchors = np.zeros((num_anchors, 4))
    # wh的值
    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T
    for i in range(len(ratios)):
        anchors[3 * i:3 * i + 3, 2] *= ratios[i][0]
        anchors[3 * i:3 * i + 3, 3] *= ratios[i][1]
    # 所有偶数列取值
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    # 所有奇数列取值
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


def shift(shape, anchors, stride=config.rpn_stride):
    """
    shift 函数的主要作用是生成一组在图像上不同位置的偏移锚框。
    这些偏移锚框将用于检测图像中不同位置的目标物体，
    以便在目标检测任务中进行训练和预测。
    这有助于模型捕捉不同位置的物体，从而提高检测的准确性。
    根据base网络的输出宽高来调整原始锚框
    """
    shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # 38, 38

    shift_x = np.reshape(shift_x, [-1])  # 1444
    shift_y = np.reshape(shift_y, [-1])  # 1444
    shifts = np.stack(
        [
            shift_x,
            shift_y,
            shift_x,
            shift_y
        ],
        axis=0  # 4x1444
    )

    shifts = np.transpose(shifts)  # 1444x4
    number_of_anchors = np.shape(anchors)[0]  # 9
    k = np.shape(shifts)[0]  # 1444
    # 1x9x4 + 1444x1x4 = 1444x9x4
    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    # 1444x9, 4
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors


def get_anchors(shape, width, height):
    anchors = generate_anchors()
    # 38x38x9, 4
    network_anchors = shift(shape, anchors)
    # 除以输入宽高来达成归一化
    network_anchors[:, 0] = network_anchors[:, 0] / width
    network_anchors[:, 1] = network_anchors[:, 1] / height
    network_anchors[:, 2] = network_anchors[:, 2] / width
    network_anchors[:, 3] = network_anchors[:, 3] / height
    # 确保值在（0，1）之间
    network_anchors = np.clip(network_anchors, 0, 1)
    return network_anchors
