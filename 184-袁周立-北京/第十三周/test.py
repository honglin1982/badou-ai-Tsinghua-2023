import numpy as np
from PIL import Image, ImageDraw

from keras.applications.imagenet_utils import preprocess_input

from config.config import Config
from nets.FasterRCNN import FasterRCNN
from utils.anchors import get_img_feature_map_size, get_anchors, calc_iou
from utils.proposal_layer import ProposalLayer


# 针对一张图，返回fasterRCNN的结果：(None, 6) 标签、置信度、坐标
def predict(fasterRCNN, proposal_layer, img_object, config):
    model_rpn = fasterRCNN.model_rpn
    model_classifier = fasterRCNN.model_classifier

    resize_w, resize_h = config.input_shape[:2]
    img = img_object.resize([resize_w, resize_h])

    img_data = np.array(img)
    img_data = np.expand_dims(img_data, 0)
    img_data = preprocess_input(img_data, mode='tf')

    rpn_result = model_rpn.predict(img_data)

    num_rois = config.num_rois
    # shape: (top_k, 4), x1,y1,w,h
    top_k_anchors = proposal_layer(rpn_result)

    # 删除预测的框中，长宽为0的部分
    delete_line = []
    for i, anchor in enumerate(top_k_anchors):
        if anchor[2] <= 0 or anchor[3] <= 0:
            delete_line.append(i)
    good_anchors = np.delete(top_k_anchors, delete_line, axis=0)
    good_anchors_num = len(good_anchors)

    # 最终分类的标签、置信度、在feature map尺寸上的坐标
    classifier_result = []

    # 先padding到num_rois的倍数，方便送入分类层
    padding_rows = num_rois - (good_anchors_num % num_rois)
    if padding_rows == num_rois:
        padding_rows = 0

    if padding_rows > 0:
        good_anchors = np.pad(good_anchors, ((0, padding_rows), (0, 0)), mode='edge')

    for j in range(good_anchors.shape[0] // num_rois):
        roi_inputs = good_anchors[j*num_rois: (j+1)*num_rois, :]
        roi_inputs = np.expand_dims(roi_inputs, 0)

        [cls, regress] = model_classifier.predict([img_data, roi_inputs])

        labels = np.argmax(cls[0], axis=1)
        confidences = np.max(cls[0], axis=1)

        for roi_index in range(num_rois):
            if j == (good_anchors.shape[0] // num_rois) - 1 and roi_index >= num_rois - padding_rows:
                continue
            label = labels[roi_index]
            if label == config.nb_class - 1:
                continue
            confidence = confidences[roi_index]
            if confidence < config.predict_confidence_threshold:
                continue
            dx, dy, dw, dh = regress[0, roi_index, 4*label: 4*(label+1)]
            dx /= config.classifier_regr_std[0]
            dy /= config.classifier_regr_std[1]
            dw /= config.classifier_regr_std[2]
            dh /= config.classifier_regr_std[3]

            x1, y1, w, h = roi_inputs[0, roi_index, :]
            old_center_x = x1 + w / 2
            old_center_y = y1 + h / 2

            center_x = old_center_x + w * dx
            center_y = old_center_y + h * dy
            w1 = w * np.exp(dw)
            h1 = h * np.exp(dh)

            coord_x1 = int(round(center_x - w1 / 2))
            coord_y1 = int(round(center_y - h1 / 2))
            coord_x2 = int(round(center_x + w1 / 2))
            coord_y2 = int(round(center_y + h1 / 2))

            classifier_result.append([label, confidence, coord_x1, coord_y1, coord_x2, coord_y2])

    if not classifier_result:
        return []
    classifier_result = np.array(classifier_result)

    # 坐标转为resize的尺寸并进行归一化
    classifier_result[:, 2:] = classifier_result[:, 2:] * config.rpn_stride
    classifier_result[:, [2, 4]] = classifier_result[:, [2, 4]] / resize_w
    classifier_result[:, [3, 5]] = classifier_result[:, [3, 5]] / resize_h
    classifier_result[:, 2:] = np.clip(classifier_result[:, 2:], 0, 1)

    nms_out = proposal_layer.predict_nms_out(classifier_result)
    return nms_out


def show(img, boxes, pred_out):
    origin_w, origin_h = img.size

    imageDraw = ImageDraw.Draw(img)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        x1, y1, x2, y2 = x1*origin_w, y1*origin_h, x2*origin_w, y2*origin_h
        imageDraw.rectangle([x1, y1, x2, y2], outline='red', width=3)
    for pred_box in pred_out:
        x1, y1, x2, y2 = pred_box[2:]
        x1, y1, x2, y2 = x1 * origin_w, y1 * origin_h, x2 * origin_w, y2 * origin_h
        imageDraw.rectangle([x1, y1, x2, y2], outline='blue', width=3)
    img.show()


def test():
    config = Config()

    annotation_path = 'data/2007_trainval.txt'
    with open(annotation_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    test_data_num = len(lines)
    print('测试数据总数: ', str(test_data_num))

    fasterRCNN = FasterRCNN(config)

    # 加载权重
    fasterRCNN.model_all.load_weights(config.init_weights_path)

    # 经过base_layer的卷积后，得到的feature_map尺寸
    img_feature_map_size = get_img_feature_map_size(config)
    # 获取最开始所有anchor的位置：左上角右下角坐标，shape:(all_anchors_num, 4)，该坐标经过了归一化
    anchors = get_anchors(img_feature_map_size, config)

    proposal_layer = ProposalLayer(anchors, img_feature_map_size, config)

    evaluate = []
    TPFPFN_record = []
    show_pic = False
    for i in range(test_data_num):
        line = lines[i].split()

        img_path = line[0]
        img = Image.open(img_path)
        boxes = np.array([list(map(int, e.split(","))) for e in line[1:]])
        boxes = np.array(boxes, dtype=np.float32)

        # boxes坐标归一化
        origin_w, origin_h = img.size
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / origin_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / origin_h

        pred_out = predict(fasterRCNN, proposal_layer, img, config)

        if show_pic:
            show(img, boxes, pred_out)

        if len(pred_out) == 0:
            evaluate.append([0, 0, 0])
            TPFPFN_record.append([0, 0, len(boxes)])
            if config.verbose:
                print("test {}/{}, precision: {}, recall: {}, F1: {}".format(i+1, test_data_num, 0, 0, 0))
            continue

        box_out_iou = np.apply_along_axis(calc_iou, 1, boxes, pred_out[:, 2:])

        # 计算precision、recall、F1
        TP = 0
        for index in range(len(pred_out)):
            flag = 0
            pred_label = pred_out[index][0]
            for box_index in range(len(boxes)):
                box_label = boxes[box_index, 4]
                # 标签相同且iou大于0.5算预测正确
                if int(pred_label) == int(box_label) and box_out_iou[box_index, index] > 0.5:
                    flag = 1
                    break
            if flag:
                TP += 1
        precision = TP / len(pred_out)
        recall = TP / len(boxes)
        F1 = 2 * precision * recall / (precision + recall + 1e-5)
        if config.verbose:
            print("test {}/{}, precision: {}, recall: {}, F1: {}".format(i+1, test_data_num, precision, recall, F1))
        evaluate.append([precision, recall, F1])
        TPFPFN_record.append([TP, len(pred_out) - TP, len(boxes) - TP])

    avg_precision = np.mean(evaluate[:, 0])
    avg_recall = np.mean(evaluate[:, 1])
    avg_f1 = np.mean(evaluate[:, 2])
    print("avg_precision: ", avg_precision)
    print("avg_recall: ", avg_recall)
    print("avg_f1: ", avg_f1)

    TP = np.sum(TPFPFN_record[:, 0])
    FP = np.sum(TPFPFN_record[:, 1])
    FN = np.sum(TPFPFN_record[:, 2])
    print("micro precision: ", TP / (TP + FP))
    print("micro recall: ", TP / (TP + FN))


if __name__ == '__main__':
    test()
