import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# 加载图片
def load_image(image_path):
    # 读取图片：rgb
    img = plt.imread(image_path)
    # 这里是从 h,w 中选出短边
    short_edge = min(img.shape[:2])  # 获取图片的短边长度
    # 将 h,w 同时减去 短边 再 /2，是为了得到长边的截取开始像素位置
    h = int((img.shape[0] - short_edge) / 2)
    w = int((img.shape[1] - short_edge) / 2)
    # 这里是对图片做切片操作，短边不变，长边从之前获取的开始位置做像素切片，和短边一样长
    crop_img = img[h: h+short_edge, w: w+short_edge]
    return crop_img


# 调整图像：默认双线性插值
def resize_image(image, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    with tf.name_scope("resize_image"):
        image = tf.expand_dims(image, 0)  # 在第一个位置插入一个维度
        # 不对齐角点以获取更平滑的结果
        image = tf.image.resize_images(image, size=size, method=method, align_corners=align_corners)
        # tf.stack 的结果为一个一维变量
        image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))
        return image


# 打印预测的结果
def print_prob(prob, file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    synset = [i.strip() for i in data]
    # 将概率按从大到小的顺序进行排列，得到结果序号
    pred = np.argsort(-prob)  # 1000类
    top5 = [print(f"top{i+1}", synset[pred[i]], prob[pred[i]]) for i in range(5)]

