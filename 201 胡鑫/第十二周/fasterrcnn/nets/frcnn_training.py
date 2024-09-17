from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from random import shuffle
from utils.anchors import get_anchors
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as k
from keras.objectives import categorical_crossentropy
import numpy as np
import random
import tensorflow as tf
import keras


def rand(a=0.0, b=1.0):
    return np.random.rand() * (b - a) + a


def cls_loss(ratio=3):
    def _cls_loss(y_true, y_pred):
        # y_true [batch_size, num_anchor, num_classes + 1]
        # y_pred [batch_size, num_anchor, num_classes]
        labels = y_true
        anchor_state = y_true[:, :, -1]  # -1是需要忽略的，0是背景，1是存在目标
        classification = y_pred

        # 找出存在目标的先验框
        # 返回一个包含满足条件的元素的索引的张量
        indices_for_object = tf.where(keras.backend.equal(anchor_state, 1))
        # 根据索引取值
        labels_for_object = tf.gather_nd(labels, indices_for_object)
        classification_for_object = tf.gather_nd(classification, indices_for_object)
        # 计算分类损失
        cls_loss_for_object = keras.backend.binary_crossentropy(labels_for_object, classification_for_object)

        # 找出实际上为背景的先验框, 计算损失
        indices_for_back = tf.where(keras.backend.equal(anchor_state, 0))
        labels_for_back = tf.gather_nd(labels, indices_for_back)
        classification_for_back = tf.gather_nd(classification, indices_for_back)

        cls_loss_for_back = keras.backend.binary_crossentropy(labels_for_back, classification_for_back)

        # 标准化，实际上是正样本的数量
        # 生成一个包含正样本锚框索引的张量
        normalizer_pos = tf.where(keras.backend.equal(anchor_state, 1))
        # 通过shape()[0]计算出所有正样本锚框的数量，然后将其转化为浮点数
        normalizer_pos = keras.backend.cast(keras.backend.shape(normalizer_pos)[0], keras.backend.floatx())
        # 确保这个浮点数至少为1.0，防止后面有除以0的情况
        normalizer_pos = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer_pos)

        # 同理，背景锚框数量
        normalizer_neg = tf.where(keras.backend.equal(anchor_state, 0))
        normalizer_neg = keras.backend.cast(keras.backend.shape(normalizer_neg)[0],
                                            keras.backend.floatx())
        normalizer_neg = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer_neg)

        # 将获得的loss除上正样本的数量
        cls_loss_for_object = keras.backend.sum(cls_loss_for_object) / normalizer_pos
        cls_loss_for_back = ratio * keras.backend.sum(cls_loss_for_back) / normalizer_neg
        # 总loss
        loss = cls_loss_for_object + cls_loss_for_back
        return loss
    return _cls_loss


def smooth_l1(sigma=1.0):
    """用于锚框回归"""
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # y_true [batch_size, num_anchor, 4+1]
        # y_pred [batch_size, num_anchor, 4]
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # 找到正样本
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算 smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma_squared
        # f(x) = |x| - 0.5 / sigma_squared    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),  # 若小于1.0 / sigma_squared
            regression_diff - 0.5 / sigma_squared  # 不小于，就使用这部分
        )

        # 如果正样本数量小于，就设置为1(防止除以0)
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        # 标准化loss
        loss = keras.backend.sum(regression_loss) / normalizer
        return loss
    return _smooth_l1


def class_loss_regr(num_classes):
    epsilon = 1e-4

    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4 * num_classes:] - y_pred
        x_abs = keras.backend.abs(x)
        x_bool = keras.backend.cast(keras.backend.less_equal(x_abs, 1.0), 'float32')
        # 这是损失的计算公式，它包含了以下几个步骤：
        # y_true[:, :, :4*num_classes] 选择了真实标签中与目标位置信息有关的部分。
        # (x_bool * (0.5*x*x) + (1+x_bool) * (x_abs-0.5)) 部分是平滑 L1 损失的计算，
        # 根据平滑 L1 损失的公式，它对每个元素计算了不同的损失。
        # 开头的4是超参数，经验，调整平滑L1损失的权重
        loss = 4 * k.sum(y_true[:, :, :4*num_classes] *
                         (x_bool * (0.5*x*x) + (1-x_bool) * (x_abs-0.5))) / k.sum(
                        epsilon + y_true[:, :, :4*num_classes])
        return loss
    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return k.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))


def get_new_img_size(width, height, img_min_side=600):
    """指定最小边为600，并保持原来的长宽比"""
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)
    return resized_width, resized_height


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [7, 3, 1, 1]
        padding = [3, 1, 0, 0]
        stride = 2
        for i in range(4):
            # input_length = (input_length - filter_size + stride) // stride
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length  # 注意缩进
    return get_output_length(width), get_output_length(height)


class Generator(object):
    def __init__(self, bbox_util, train_lines, num_classes, solid, solid_shape=[600, 600]):
        self.bbox_util = bbox_util
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.num_classes = num_classes
        self.solid = solid
        self.solid_shape = solid_shape

    def get_random_data(self, annotation_line, jitter=.1, hue=.1, sat=1.1, val=1.1):
        """实时数据增强的随机预处理"""
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        if self.solid:
            w, h = self.solid_shape
        else:
            w, h = get_new_img_size(iw, ih)

        # 将line[1:]中的每个标注框信息转换为box数组，其中每个标注框由四个坐标值表示（左上角x、左上角y、右下角x、右下角y）。
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # resize image(微小缩放)
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(.9, 1.1)

        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image (图象粘贴)
        # 缩放后的图象在原始图象中的偏移
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        image_new = Image.new('RGB', (w, h), (128, 128, 128))  # 第三个参数表示颜色，这里是灰色
        # 将缩放后的图象粘贴到原始图象上
        image_new.paste(image, (dx, dy))
        image = image_new

        # flip image or not(是否翻转)
        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image(扭曲图象)
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        # 三个通道色度调整
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image_data = hsv_to_rgb(x) * 255

        # correct boxes (基于上面的对图象处理，来对框进行处理)
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            # 将标注框的水平坐标进行缩放，并加上偏移量dx
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            # 将标注框的垂直坐标进行缩放，并加上偏移量dy
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                # 将标注框的水平坐标进行翻转(满足条件的情况下)
                box[:, [0, 2]] = w - box[:, [2, 0]]
            # 对标注框进行了边界限制
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box(丢弃无效框)
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []
        if (box_data[:, :4] > 0).any():  # 有任意一个值大于0
            return image_data, box_data
        else:
            return image_data, []

    def generate(self):
        while True:
            shuffle(self.train_lines)
            lines = self.train_lines
            for annotation_line in lines:
                img, y = self.get_random_data(annotation_line)
                height, width, _ = np.shape(img)

                if len(y) == 0:  # 如果是无效框就继续下一个
                    continue
                boxes = np.array(y[:, :4], dtype=np.float32)
                boxes[:, 0] = boxes[:, 0] / width
                boxes[:, 1] = boxes[:, 1] / height
                boxes[:, 2] = boxes[:, 2] / width
                boxes[:, 3] = boxes[:, 3] / height

                box_heights = boxes[:, 3] - boxes[:, 1]
                box_widths = boxes[:, 2] - boxes[:, 0]

                if (box_heights <= 0).any() or (box_widths <= 0).any():  # 有任意一个值小于等于零
                    continue

                y[:, :4] = boxes[:, :4]

                anchors = get_anchors(get_img_output_length(width, height), width, height)

                # 计算真实框对应的先验框，筛选出了-1背景框，1有效框，0忽略框
                assignment = self.bbox_util.assign_boxes(y, anchors)

                num_regions = 256
                # 上面的三个值，表示哪种框
                classification = assignment[:, 4]

                regression = assignment[:, :]
                # 有效狂的mask
                mask_pos = classification[:] > 0
                # 有效框个数
                num_pos = len(classification[mask_pos])
                # 样本平衡，正样本最多占一半
                if num_pos > num_regions / 2:
                    # 超出的部分，随机生成在有效框里的索引
                    val_locs = random.sample(range(num_pos), int(num_pos - num_regions / 2))
                    # 把在正样本中的这些随机索引所在的值变成-1，标记为背景框
                    classification[mask_pos][val_locs] = -1
                    regression[mask_pos][val_locs, -1] = -1

                # 无效框
                mask_neg = classification[:] == 0
                num_neg = len(classification[mask_neg])
                # 也是样本平衡
                if num_neg + num_pos > num_regions:
                    val_locs = random.sample(range(num_neg), int(num_neg - num_pos))
                    # 把多出来的负样本变成背景
                    classification[mask_neg][val_locs] = -1

                classification = np.reshape(classification, [-1, 1])
                # [12996, 5] 包含了三种框
                regression = np.reshape(regression, [-1, 5])

                tmp_inp = np.array(img)
                tmp_targets = [np.expand_dims(np.array(classification, dtype=np.float32), 0),  # [1,12996,1]
                               np.expand_dims(np.array(regression, dtype=np.float32), 0)]  # [1,12996,5]

                yield preprocess_input(np.expand_dims(tmp_inp, 0)), tmp_targets, np.expand_dims(y, 0)

