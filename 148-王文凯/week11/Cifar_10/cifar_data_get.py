import tensorflow as tf
import os

num_classes = 10
num_example_for_train = 50000
num_example_for_test = 10000

class DataObject(object):
    pass

def cifar_10_read(file_queue):
    res = DataObject()

    # 标签的字节数 如果是cifar_100 则字节数为2
    label_bytes = 1
    # 图像的高，宽，通道数
    res.height = 32
    res.width = 32
    res.depth = 3

    # 计算图像的总字节数与每个样本的字节数
    image_bytes = res.height * res.width * res.depth
    # 每个样本的字节数 = 图像字节数 + 标签字节数
    record_bytes = image_bytes + label_bytes

    # 使用固定长度记录一个文件读取类
    file_reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    res.key, value = file_reader.read(file_queue)

    # 将读取的文件内容从字符串形式解析为图像对应的数组
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 解析的像素数组中 第一个元素为 标签 取出并转换为int32的数值形式
    res.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 将图像数据从一维数组形式转变为三维数组形式(depth * height * weight) CHW
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [res.depth, res.height, res.width])
    # CHW -> HWC
    res.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return res

def inputs(data_dir, batch_size, distorted=None):
    file_names = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
    file_queue = tf.train.string_input_producer(file_names)
    file_input = cifar_10_read(file_queue)

    # 将图像数据转换为float32格式
    reshaped_image = tf.cast(file_input.uint8image, tf.float32)

    num_examples = num_example_for_train

    # distorted 不为 None 则将图像数据进行图像增强
    if distorted:
        # 随机剪裁
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
        # 随机左右翻转
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        # 随机亮度调整
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)
        # 随机对比度调整
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
        # 标准化图像数据
        float_image = tf.image.per_image_standardization(adjusted_contrast)

        # 设置图像数据形状
        float_image.set_shape([24, 24, 3])
        file_input.label.set_shape([1])

        min_queue_examples = int(num_examples * 0.4)
        # 使用 tf.train.shuffle_batch 创建一个随机批量的图像和标签
        image_train, label_train = tf.train.shuffle_batch([float_image, file_input.label],
                                                          batch_size=batch_size, num_threads=16,
                                                          capacity=min_queue_examples + 3 * batch_size,
                                                          min_after_dequeue=min_queue_examples,
                                                          )

        return image_train, tf.reshape(label_train, [batch_size])

    else:
        # 裁剪图像大小
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        # 标准化图像数据
        float_image = tf.image.per_image_standardization(resized_image)

        float_image.set_shape([24, 24, 3])
        file_input.label.set_shape([1])

        min_queue_examples = int(num_examples * 0.4)

        image_test, label_test = tf.train.batch([float_image, file_input.label],
                                                batch_size=batch_size, num_threads=16,
                                                capacity=min_queue_examples + 3 * batch_size
                                                )

        return image_test, tf.reshape(label_test, [batch_size])

