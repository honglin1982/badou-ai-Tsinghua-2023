import numpy as np
import math
import pdb
import copy
import time

def union(au, bu, areaIntersection):

    areaA = (au[2] - au[0]) * (au[3] - au[1])
    areaB = (bu[2] - bu[0]) * (bu[3] - bu[1])
    areaUnion = areaA + areaB - areaIntersection
    return areaUnion

def intersection(ai, bi):

    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    return 0 if w < 0 or h < 0 else w * h

def iou(a, b):

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3] : return 0.0
    areaI = intersection(a, b)
    areaU = union(a, b, areaI)
    return float(areaI) / float(areaU + 1e-6)

def calIou(R, config, allBoxes, width, height, numClasses):

    bboxes = allBoxes[:,:4]
    gta = np.zeros((len(bboxes), 4))
    for bboxNum, bbox in enumerate(bboxes):
        gta[bboxNum, 0] = int(round(bbox[0] * width  / config.rpnStride))
        gta[bboxNum, 1] = int(round(bbox[1] * height / config.rpnStride))
        gta[bboxNum, 2] = int(round(bbox[2] * width  / config.rpnStride))
        gta[bboxNum, 3] = int(round(bbox[3] * height / config.rpnStride))
    xRoi = []
    yClassNum = []
    yClassRegrCoords = []
    yClassRegrLabels = []
    IoUs = []
    for ix in range(R.shape[0]):

        x1 = R[ix, 0] * width  / config.rpnStride
        y1 = R[ix, 1] * height / config.rpnStride
        x2 = R[ix, 2] * width / config.rpnStride
        y2 = R[ix, 3] * height / config.rpnStride

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        bestIou = 0.0
        bestbbox = -1
        for bboxNum in range(len(bboxes)):
            currIou = iou([gta[bboxNum, 0], gta[bboxNum, 1],
                           gta[bboxNum, 2], gta[bboxNum, 3]],
                          [x1, y1, x2, y2])
            if currIou > bestIou :
                bestIou = currIou
                bestbbox = bboxNum
        if bestIou < config.classifierMinOverlap : continue
        else:
              w = x2 - x1
              h = y2 - y1
              xRoi.append([x1, y1, w, h])
              IoUs.append(bestIou)
              if config.classifierMinOverlap <= bestIou < config.classifierMaxOverlap : label = -1
              elif config.classifierMaxOverlap <= bestIou :

                  label = int(allBoxes[bestbbox, -1])
                  cxg = (gta[bestbbox, 0] + gta[bestbbox, 2]) / 2.0
                  cyg = (gta[bestbbox, 1] + gta[bestbbox, 3]) / 2.0

                  cx = x1 + w / 2.0
                  cy = y1 + h / 2.0
                  tx = (cxg - cx) / float(w)
                  ty = (cyg - cy) / float(h)
                  tw = np.log((gta[bestbbox, 2] - gta[bestbbox, 0]) / float(w))
                  th = np.log((gta[bestbbox, 3] - gta[bestbbox, 1]) / float(h))
              else :
                  print('roi = {}'.format(bestIou))
                  raise RuntimeError
        classLabel = numClasses * [0]
        classLabel[label] = 1
        yClassNum.append(copy.deepcopy(classLabel))
        coords = [0] * 4 * (numClasses - 1)
        labels = [0] * 4 * (numClasses - 1)
        if label != -1:

            labelPos = 4 * label
            sx, sy, sw, sh = config.classifierRegrStd
            coords[labelPos:4 + labelPos] = [sx * tx, sy * ty, sw * tw, sh * th]
            labels[labelPos:4 + labelPos] = [1, 1, 1, 1]
            yClassRegrCoords.append(copy.deepcopy(coords))
            yClassRegrLabels.append(copy.deepcopy(labels))
        else :
            yClassRegrCoords.append(copy.deepcopy(coords))
            yClassRegrLabels.append(copy.deepcopy(labels))

        if len(xRoi) == 0: return None, None, None, None
        X = np.array(xRoi)
        Y1 = np.array(yClassNum)
        Y2 = np.concatenate([np.array(yClassRegrLabels), np.array(yClassRegrCoords)],axis=1)
        return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs
