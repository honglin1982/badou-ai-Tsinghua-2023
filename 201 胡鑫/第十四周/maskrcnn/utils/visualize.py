"""
mask r-cnn
可视化函数
"""
import os
import sys
import random
import itertools
import colorsys
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon

# 项目根目录
ROOT_DIR = os.path.abspath('../')

# 查找库的本地版本
sys.path.append(ROOT_DIR)
from utils import utils


############################################################
#  Visualization
############################################################
def random_colors(N, bright=True):
    """
    生成随机颜色。
    要获得视觉上不同的颜色，请在HSV空间中生成它们，然后转换为RGB。
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """将给定的mask应用于图像。"""
    # 将mask==1的位置变色
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names, scores=None,
                      title="", figsize=(16, 16), ax=None, show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    masks：[height, width, num_instances]
    class_ids:[num_instances]
    classname：数据集的类名列表
    scores：（可选）每个框的置信度分数
    title：（可选）图形标题
    show_mask，show_box:是否显示遮罩和边界框
    figsize：（可选）图像的大小
    colors：（可选）与每个对象一起使用的一个或多个颜色
    captures：（可选）用作每个对象的标题的字符串列表
    """
    # 实例个数
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # 如果没有传递轴，请创建一个轴并自动调用show（）
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # 随机颜色
    colors = colors or random_colors(N)

    # 显示图象边界外的区域
    height, width = image.shape[:2]
    # 设置坐标轴展示范围
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()

    for i in range(N):
        color = colors[i]

        # 边界框
        if not np.any(boxes[i]):
            # 跳过此实例。没有bbox。可能在图像裁剪中丢失。
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption, color='w', size=11, backgroundcolor='none')

        # mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # mask多边形
        # 填充以确保接触图像边缘的mask具有正确的多边形。
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        # 查找padded_mask中的轮廓信息
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # 遍历每个找到的轮廓信息，减去填充并将（y，x）翻转为（x，y），-1是为了矫正坐标的绘制位置
            # 某些图形库可能从0开始索引坐标
            verts = np.fliplr(verts) - 1
            # 创建了一个多边形对象p，它使用翻转后的坐标 verts 表示轮廓。facecolor='none' 表示多边形的填充颜色为空，
            # 即透明，edgecolor=color 设置多边形的边缘颜色为指定的颜色
            p = Polygon(verts, facecolor='none', edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
