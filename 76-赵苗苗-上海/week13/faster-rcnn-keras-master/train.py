from __future__ import division
from nets.frcnn import get_model
from nets.frcnn_training import cls_loss,smooth_l1,Generator,get_img_output_length,class_loss_cls,class_loss_regr

from utils.config import Config
from utils.utils import BBoxUtility
from utils.roi_helpers import calc_iou

from keras.utils import generic_utils
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import keras
import numpy as np
import time 
import tensorflow as tf
from utils.anchors import get_anchors

#将训练过程中的日志信息写入TensorBoard中
def write_log(callback, names, logs, batch_no):#callback(TensorBoard回调对象)，names日志名称列表，logs日志值列表，batch_no当前批次的序号
    for name, value in zip(names, logs): #同时遍历名称列表和日志值列表，以便逐个将它们配对起来
        summary = tf.Summary()   #创建Summary对象，用于保存单个日志项的摘要信息
        summary_value = summary.value.add()   
        summary_value.simple_value = value    #为summary.value添加一个元素，并将对应的日志值赋给simple_value属性
        summary_value.tag = name   #将日志的名称赋给tag属性
        callback.writer.add_summary(summary, batch_no)  #将该日志摘要添加到TensorBoard中
        callback.writer.flush()   #确保将日志写入到TensorBoard中

if __name__ == "__main__":
    #创建配置对象config，设置NUM_CLASSES总类别数，EPOCH总轮次，EPOCH_LENGTH每轮次的步数，实例化BBoxUtility类，传入RPN最大重叠和忽略的阈值作为参数，设定标注数据文件路径annotation_path
    config = Config()
    NUM_CLASSES = 21
    EPOCH = 100
    EPOCH_LENGTH = 2000
    bbox_util = BBoxUtility(overlap_threshold=config.rpn_max_overlap,ignore_threshold=config.rpn_min_overlap)
    annotation_path = '2007_train.txt'
    
    #调用get_model函数创建RPN模型model_rpn，分类器模型model_classifier，整体模型model_all，加载预训练权重base_net_weights到RPN模型和分类器模型中
    model_rpn, model_classifier,model_all = get_model(config,NUM_CLASSES)
    base_net_weights = "model_data/voc_weights.h5"
    model_all.summary()
    model_rpn.load_weights(base_net_weights,by_name=True)
    model_classifier.load_weights(base_net_weights,by_name=True)

    #打开标注数据文件，将每行数据读取列表lines中，其中每行表示一个标注样本，使用随机种子对lines进行随机打乱
    with open(annotation_path) as f: 
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    #实例化Generator类对象gen，传入参数，调用生成器对象的generate方法获取训练数据的迭代器rpn_train。创建TensorBoard回调函数logging，并将其设置为回调函数callback关联的模型
    gen = Generator(bbox_util, lines, NUM_CLASSES, solid=True)
    rpn_train = gen.generate()
    log_dir = "logs"
    logging = TensorBoard(log_dir=log_dir)
    callback = logging
    callback.set_model(model_all)

    #编译RPN模型和分类器模型，指定损失函数及优化器，整体模型不需要训练，只需在后续的训练过程中加载预训练的权重
    model_rpn.compile(loss={
                'regression'    : smooth_l1(),
                'classification': cls_loss()
            },optimizer=keras.optimizers.Adam(lr=1e-5)
    )
    model_classifier.compile(loss=[
        class_loss_cls, 
        class_loss_regr(NUM_CLASSES-1)
        ], 
        metrics={'dense_class_{}'.format(NUM_CLASSES): 'accuracy'},optimizer=keras.optimizers.Adam(lr=1e-5)
    )
    model_all.compile(optimizer='sgd', loss='mae')

    """进行一些参数的初始化，包括迭代次数，训练步数，损失值，准确率等相关变量，然后根据设置的总轮次EPOCH进行循环训练"""
    # 初始化训练过程中使用的各种参数和变量，包括迭代次数，训练步数，损失值数组，RPN准确率相关指标，记录开始训练的时间，并初始化最佳损失为无穷大
    iter_num = 0
    train_step = 0
    losses = np.zeros((EPOCH_LENGTH, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = [] 
    start_time = time.time()
    best_loss = np.Inf
    print('Starting training')

    #对总轮次EPOCH进行循环训练
    for i in range(EPOCH):
        #在第20轮时，降低学习率LR，并重新编译RPN模型和分类器模型
        if i == 20:
            model_rpn.compile(loss={
                        'regression'    : smooth_l1(),
                        'classification': cls_loss()
                    },optimizer=keras.optimizers.Adam(lr=1e-6)
            )
            model_classifier.compile(loss=[
                class_loss_cls, 
                class_loss_regr(NUM_CLASSES-1)
                ], 
                metrics={'dense_class_{}'.format(NUM_CLASSES): 'accuracy'},optimizer=keras.optimizers.Adam(lr=1e-6)
            )
            print("Learning rate decrease")
        
        #创建用于显示进度的Progbar对象，并打印当前训练轮次
        progbar = generic_utils.Progbar(EPOCH_LENGTH) 
        print('Epoch {}/{}'.format(i + 1, EPOCH))
        
        #开始当前轮次的具体训练步骤
        while True:
            #检查并打印RPN准确率相关指标，包括平均重叠的边界框数
            if len(rpn_accuracy_rpn_monitor) == EPOCH_LENGTH and config.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, EPOCH_LENGTH))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
            
            #从生成器中获取下一批训练数据，计算RPN网络的损失值，并使用模型训练每个batch的数据，通过RPN模型预测边界框得到P_rpn、计算先验框anchors和预测结果results，并将预测结果解码
            X, Y, boxes = next(rpn_train)
            loss_rpn = model_rpn.train_on_batch(X,Y)
            write_log(callback, ['rpn_cls_loss', 'rpn_reg_loss'], loss_rpn, train_step)
            P_rpn = model_rpn.predict_on_batch(X)
            height,width,_ = np.shape(X[0])
            anchors = get_anchors(get_img_output_length(width,height),width,height)
            results = bbox_util.detection_out(P_rpn,anchors,1, confidence_threshold=0)
            R = results[0][:, 2:]
            X2, Y1, Y2, IouS = calc_iou(R, config, boxes[0], width, height, NUM_CLASSES)

            #如果未检测到正样本，记录RPN准确率指标为0，并跳过当前batch的训练
            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue
            
            #根据标签数据Y1，将负样本和正样本存储在neg_samples和pos_samples中
            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)
            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []
            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []
            
            #记录该batch的正样本数量到rpn_accuracy_rpn_monitor中，并将该数量添加到每轮结束时的rpn_accuracy_for_epoch列表中，如果未找到负样本，跳过当前batch的训练
            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))
            if len(neg_samples)==0:
                continue
            
            #根据正负样本的数量，从列表中选择用于训练的正负样本，如果正样本数量小于每个batch所需的一半(config.num_rois//2)，则将所有正样本都选中，否则，从正样本中随机选择一半的样本，
            #然后，从负样本中随机选择剩余数量的样本，将选中的样本保存在sel_samples列表中，并使用这些样本来训练分类器模型
            if len(pos_samples) < config.num_rois//2:
                selected_pos_samples = pos_samples.tolist()
            else:
                selected_pos_samples = np.random.choice(pos_samples, config.num_rois//2, replace=False).tolist()
            try:
                selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples), replace=False).tolist()
            except:
                selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples), replace=True).tolist()
            
            sel_samples = selected_pos_samples + selected_neg_samples
            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
            
            #将分类器模型的损失值写入日志文件
            write_log(callback, ['detection_cls_loss', 'detection_reg_loss', 'detection_acc'], loss_class, train_step)

            #将当前batch的损失值记录在数组losses中，然后更新迭代次数iter_num和训练步数train_step，使用progbar对象更新训练进度
            losses[iter_num, 0]  = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]
            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1
            train_step += 1
            progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

            #计算每轮结束时的平均损失值：RPN分类器的平均分类损失，RPN回归器的平均回归损失，检测器分类器的平均分类损失，
            if iter_num == EPOCH_LENGTH:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []
                if config.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                #总损失值计算：RPN分类器损失值+RPN回归器损失值+检测器分类器损失值+检测器回归器损失值
                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()
                
                #训练统计信息写入日志文件，记录已用时间，平均重叠的边界框数，RPN分类器和回归器的平均损失值、检测器分类器和回归器的平均损失值，检测器分类准确率以及总损失值
                write_log(callback,
                        ['Elapsed_time', 'mean_overlapping_bboxes', 'mean_rpn_cls_loss', 'mean_rpn_reg_loss',
                        'mean_detection_cls_loss', 'mean_detection_reg_loss', 'mean_detection_acc', 'total_loss'],
                        [time.time() - start_time, mean_overlapping_bboxes, loss_rpn_cls, loss_rpn_regr,
                        loss_class_cls, loss_class_regr, class_acc, curr_loss],i)
                    
                #最佳损失值更新和模型权重保存
                if config.verbose:  #如果设定了输出详细信息，则执行以下代码
                    print('The best loss is {}. The current loss is {}. Saving weights'.format(best_loss,curr_loss))
                if curr_loss < best_loss:
                    best_loss = curr_loss  
                model_all.save_weights(log_dir+"/epoch{:03d}-loss{:.3f}-rpn{:.3f}-roi{:.3f}".format(i,curr_loss,loss_rpn_cls+loss_rpn_regr,loss_class_cls+loss_class_regr)+".h5")
                #跳出循环
                break
