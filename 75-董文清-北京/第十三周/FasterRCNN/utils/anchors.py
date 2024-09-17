import keras.backend
import numpy as np
from .config import Config

config = Config()
def generateAnchors(sizes=None, ratios=None):

    sizes = config.anchorBoxScales if sizes is None else sizes
    ratios = config.anchorBoxRatios if ratios is None else ratios

    numAnchors = len(sizes) * len(ratios)
    anchors = np.zeros((numAnchors, 4))
    #复制anchors
    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T
    for i in range(len(ratios)):
        anchors[3 * i : 3 * i + 3, 2] = anchors[3 * i : 3 * i + 3, 2] * ratios[i][0]
        anchors[3 * i : 3 * i + 3, 3] = anchors[3 * i : 3 * i + 3, 3] * ratios[i][1]

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors

def shift(shape, anchors, stride=config.rpnStride):

    shiftX = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
    shiftY = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride

    #构造以shiftX、shiftY为轴坐标的网格
    shiftX, shiftY = np.meshgrid(shiftX, shiftY)
    #都降成一维数组
    shiftX, shiftY = np.reshape(shiftX, [-1]), np.reshape(shiftY, [-1])

    shifts = np.stack([shiftX, shiftY, shiftX, shiftY], axis=0)
    shifts = np.transpose(shifts)
    numberOfAnchors = np.shape(anchors)[0]
    k = np.shape(shifts)[0]
    shiftedAnchors = np.reshape(anchors, [1, numberOfAnchors, 4]) + \
                     np.array(np.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shiftedAnchors = np.reshape(shiftedAnchors, [k * numberOfAnchors, 4])
    return shiftedAnchors

def getAnchors(shape, width, height):

    anchors = generateAnchors()
    networkAnchors = shift(shape, anchors)
    networkAnchors[:,0] = networkAnchors[:,0]/width
    networkAnchors[:,1] = networkAnchors[:,1]/height
    networkAnchors[:,2] = networkAnchors[:,2]/width
    networkAnchors[:,3] = networkAnchors[:,3]/height
    #将anchors 值 缩小至 [0, 1]
    networkAnchors = np.clip(networkAnchors, 0, 1)
    return networkAnchors
