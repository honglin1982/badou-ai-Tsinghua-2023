import os.path
import numpy as np
import tensorflow as tf


class CIFAR10Record(object):
    pass


def read_cifar10(file_queue):
    result = CIFAR10Record()

    result.height = 32
    result.width = 32
    result.channel = 3
    label_bytes = 1
    # 图片样本元素个数
    image_bytes = result.height * result.width * result.channel
    # 加label总个数
    record_bytes = image_bytes + label_bytes
    # 按固定长度读取文件
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # 读取文件
    result.key, value = reader.read(file_queue)
    # 读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组
    record_bytes = tf.decode_raw(value, tf.uint8)
    # 提取标签 并且转换为int32
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), np.int32)
    # 剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
    # 这一步是将一维数据转换成3维数据
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.channel, result.height, result.width])
    # 这一步是转换数据排布方式，变为(h,w,c)
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


def inputs(data_dir, batch_size, distorted):
    fileNames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
    file_queue = tf.train.string_input_producer(fileNames)
    read_input = read_cifar10(file_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    num_examples_per_epoch = 50000
    # 根据入参判断是否需要图像增强
    if distorted is not None:
        # 对图像进行随机裁剪
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
        # 将剪切好的图片进行左右翻转
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        # 将左右翻转好的图片进行随机亮度调整
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)
        # 将亮度调整好的图片进行随机对比度调整
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
        # 图片进行标准化处理（归一化）
        float_image = tf.image.per_image_standardization(adjusted_contrast)
        # 设置图片数据及标签的形状
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_examples = int(10000 * 0.4)
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label],
                                                            batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue=min_queue_examples)
        session = tf.Session()
        n_images_train = session.run(images_train)
        return images_train, tf.reshape(labels_train, [batch_size])
    else:

        # 在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        # 剪切完成以后，直接进行图片标准化操作
        float_image = tf.image.per_image_standardization(resized_image)

        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_per_epoch * 0.4)

        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size,
                                                  num_threads=16,
                                                  capacity=min_queue_examples + 3 * batch_size)
        # 这里使用batch()函数代替tf.train.shuffle_batch()函数
        return images_test, tf.reshape(labels_test, [batch_size])
