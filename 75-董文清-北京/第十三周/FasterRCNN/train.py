from __future__ import division
from nets.frcnn import  getModel
from nets.frcnnTraining import clsLoss, smoothL1, Generator, \
    getImgOutputLength, classLossCls, classLossRegr
from utils.config import Config
from utils.utils import BBoxUtility
from utils.roiHelpers import  calIou
from keras.utils import generic_utils
from keras.callbacks import TensorBoard
import keras
import numpy as np
import time
import tensorflow as tf
from utils.anchors import getAnchors

def writeLog(callback, names, logs, batchNo):

    for name, value in zip(names, logs):

        summary = tf.Summary()
        summaryValue = summary.value.add()
        summaryValue.simple_value = value
        summaryValue.tag = name
        callback.writer.add_summary(summary, batchNo)
        callback.writer.flush()

if __name__ == '__main__':

    config = Config()
    numClasses = 21
    epoch = 100
    epochLength = 2000
    bboxUtil = BBoxUtility(overlapThrehold=config.rpnMaxOverlap, ignoreThreshold=config.rpnMinOverlap)
    annotationPath = '2007_train.txt'

    modelRpn, modelClassifier, modelAll = getModel(config, numClasses)
    baseNetWeights = "model_data/voc_weights.h5"
    modelAll.summary()
    modelRpn.load_weights(baseNetWeights,by_name=True)
    modelClassifier.load_weights(baseNetWeights, by_name=True)

    #随机
    with open(annotationPath) as f : lines = f.readlines()
    np.random.seed(19392)
    np.random.shuffle(lines)
    np.random.seed(None)
    gen = Generator(bboxUtil, lines, numClasses, solid=True)
    rpnTrain = gen.generate()
    logDir = "logs"
    #训练参数设置
    logging = TensorBoard(log_dir=logDir)
    callback = logging
    callback.set_model(modelAll)

    modelRpn.compile(loss={

        'regression': smoothL1(),
        'classification': clsLoss()
    }, optimizer=keras.optimizers.Adam(lr=1e-5))
    #metrics 评价指标函数
    modelClassifier.compile(loss=[classLossCls, classLossRegr(numClasses-1)],
                            metrics={'dense_class_{}'.format(numClasses): 'accuracy'},
                            optimizer=keras.optimizers.adam(lr=1e-5))
    modelAll.compile(optimizer='sgd', loss='mae')

    #参数初始化
    iterNum = 0
    trainStep = 0
    losses = np.zeros((epochLength, 5))
    rpnAccuracyRpnMonitor = []
    rpnAccuracyForEpoch = []
    startTime = time.time()
    #最佳loss
    bestLoss = np.Inf
    print('Starting training')
    for i in range(epoch):

        if i == 20:
            #第二十轮次时 调整学习率到1e-6
            modelRpn.compile(loss={

                'regression' : smoothL1(),
                'classification' : clsLoss()
            }, optimizer=keras.optimizers.Adam(lr=1e-6))
            modelClassifier.compile(loss=[
                classLossCls,
                classLossRegr(numClasses-1)
            ], metrics={'dense_class_{}'.format(numClasses): 'accuracy'},
               optimizer=keras.optimizers.Adam(lr=1e-6))
            print('Learning rate decrease')
        probar = generic_utils.Progbar(epochLength)
        print('Epoch {}/{}'.format(i + 1, epoch))
        while True:

            if len(rpnAccuracyRpnMonitor) == epochLength and config.verbose:
                meanOverlappingBBoxes = float(sum(rpnAccuracyRpnMonitor)) / len(rpnAccuracyRpnMonitor)
                rpnAccuracyRpnMonitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(meanOverlappingBBoxes, epochLength))
                if meanOverlappingBBoxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            X, Y, boxes = next(rpnTrain)
            lossRpn = modelRpn.train_on_batch(X, Y)
            writeLog(callback, ['rpn_cls_loss', 'rpn_reg_loss'], lossRpn, trainStep)
            Prpn = modelRpn.predict_on_batch(X)
            height, width = np.shape(X[0])
            anchors = getAnchors(getImgOutputLength(width, height), width, height)

            #将预测结果进行解码
            results = bboxUtil.detectionOut(Prpn, anchors, 1, confidenceThreshold=0)
            R = results[0][:, 2:]
            X2, Y1, Y2, Ious = calIou(R, config, boxes[0], width, height, numClasses)
            if X2 is None :
                  rpnAccuracyRpnMonitor.append(0)
                  rpnAccuracyForEpoch.append(0)
                  continue
            negSamples = np.where(Y1[0, :, -1] == 1)
            posSamples = np.where(Y1[0, :, -1] == 0)

            negSamples = negSamples[0] if len(negSamples) > 0 else []
            posSamples = posSamples[0] if len(posSamples) > 0 else []

            rpnAccuracyRpnMonitor.append(len(posSamples))
            rpnAccuracyForEpoch.append(len(posSamples))
            if len(negSamples) == 0 : continue

            if len(posSamples) < config.numRois // 2: selectedPosSamples = posSamples.tolist()
            else : selectedPosSamples = np.random.choice(negSamples, config.numRois//2, replace=False).tolist()
            try: selectedNegSamples = np.random.choice(negSamples, config.numRois - len(selectedNegSamples), replace=False).tolist()
            except : selectedNegSamples = np.random.choice(negSamples, config.numRois - len(selectedNegSamples), replace=True).tolist()
            selSamples = selectedPosSamples + selectedNegSamples
            lossClass = modelClassifier.train_on_batch([X, X2[:, selSamples, :]], [Y1[:,selSamples,:], Y2[:,selSamples,:]])
            writeLog(callback, ['detection_cls_loss', 'detection_reg_loss', 'detection_acc'], lossClass, trainStep)

            losses[iterNum, 0] = lossRpn[1]
            losses[iterNum, 1] = lossRpn[2]
            losses[iterNum, 2] = lossClass[1]
            losses[iterNum, 3] = lossClass[2]
            losses[iterNum, 4] = lossClass[3]
            trainStep += 1
            iterNum += 1
            probar.update(iterNum, [('rpn_cls', np.mean(losses[:iterNum,0])), ('rpn_regr', np.mean(losses[:iterNum, 1])),
                                    ('detector_cls', np.mean(losses[:iterNum, 2])), ('detector_regr', np.mean(losses[:iterNum, 3]))])
            if iterNum == epochLength:

                lossRpnCls = np.mean(losses[:, 0])
                lossRpnRegr = np.mean(losses[:, 1])
                lossClassCls = np.mean(losses[:, 2])
                lossClassRegr = np.mean(losses[:, 3])
                classAcc = np.mean(losses[:,4])
                meanOverlappingBBoxes = float(sum(rpnAccuracyForEpoch)) / len(rpnAccuracyForEpoch)
                rpnAccuracyForEpoch = []

                if config.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(meanOverlappingBBoxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(classAcc))
                    print('Loss RPN classifier: {}'.format(lossRpnCls))
                    print('Loss RPN regression: {}'.format(lossRpnRegr))
                    print('Loss Detector classifier: {}'.format(lossClassCls))
                    print('Loss Detector regression: {}'.format(lossRpnRegr))
                    print('Elapsed time: {}'.format(time.time() - startTime))

                curLoss = lossRpnCls + lossRpnRegr + lossClassCls + lossClassRegr
                iterNum = 0
                startTime = time.time()
                writeLog(callback, ['Elapsed_time', 'mean_overlapping_bboxes', 'mean_rpn_cls_loss', 'mean_rpn_reg_loss',
                        'mean_detection_cls_loss', 'mean_detection_reg_loss', 'mean_detection_acc', 'total_loss'],
                         [time.time() - startTime, meanOverlappingBBoxes, lossRpnCls, lossRpnRegr,
                          lossClassCls, lossClassRegr, classAcc, curLoss], i)

                if config.verbose:
                    print('The best loss is {}. The current loss is {}. Saving weights'.format(bestLoss, curLoss))
                if curLoss < bestLoss :
                    bestLoss = curLoss
                modelAll.save_weights(logDir + "/epoch{:03d}-loss{:.3f}-rpn{:.3f}-roi{:.3f}".format(i, curLoss, lossRpnCls + lossRpnRegr, lossClassCls + lossClassRegr)+".h5")
                break