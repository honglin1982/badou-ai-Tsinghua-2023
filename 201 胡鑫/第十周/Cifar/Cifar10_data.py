# 该文件负责读取Cifar-10数据并对其进行数据增强预处理
import os
import tensorflow as tf

num_classes = 10

#  设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000


# 定义一个空类，用于返回读取的Cifar-10的数据
class CIFAR10Record(object):
    pass


# 定义一个读取Cifar-10的函数，这个函数的目的就是读取目标文件的数据
def read_cifar10(file_queue):
    result = CIFAR10Record()
    # 如果是cifar-100数据集，则此处为2
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3  # 因为是rgb图像，所以深度是3

    # 图片样本总元素个数
    img_bytes = result.height * result.width * result.depth
    # 所有总的个数
    record_bytes = label_bytes + img_bytes

    # 使用tf.FixedLengthRecordReader()创建一个文件读取类。该类的目的就是读取文件
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # 读取文件
    result.key, value = reader.read(file_queue)
    # 读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组
    record_bytes = tf.decode_raw(value, tf.uint8)
    # 因为该数组第一个元素是标签，所以我们使用strided_slice()函数将标签提取出来，
    # 并且使用tf.cast()函数将这一个标签转换成int32的数值形式
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 剩下的元素再分割出来就是图片数据，而在数据集中储存方式为depth，需要将其转化为
    # [depth, height, width]，这一步是将一维数据转化为三维数据
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + img_bytes]),
                             [result.depth, result.height, result.width])

    # 使用tf.transpose将[c, h, w]转换成[h, w, c]
    result.uint8img = tf.transpose(depth_major, [1, 2, 0])

    return result


# 对图像进行预处理，对图像是否进行增强进行判断，做出相应的操作
def inputs(data_dir, batch_size, distorted):
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
    # 根据已有的文件名创建一个文件队列
    file_queue = tf.train.string_input_producer(filenames)
    # 根据已经有的文件队列，使用自定义的read_cifar10()读取队列中的文件
    read_input = read_cifar10(file_queue)

    # 转换成float32
    reshaped_image = tf.cast(read_input.uint8img, tf.float32)

    num_examples_pre_epoch = num_examples_pre_epoch_for_train

    if distorted is not None:  # 如果这个值不为空，则表示要进行图像增强操作
        # 1.将图像剪切成24*24*3
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
        # 2.将剪切好的图像左右翻转
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        # 3.将左右翻转后的图像进行随机亮度调整
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)
        # 4.将亮度调整好的图片进行随机对比度调整
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
        # 5.进行标准化图片操作，对每一个像素减去平均值并除以像素方差
        float_image = tf.image.per_image_standardization(adjusted_contrast)

        # 设置图片数据及标签形状
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        print("在开始训练前用%d个CIFAR图像填充队列，这将需要几分钟时间" % min_queue_examples)

        # 使用tf.train.shuffle_batch()函数随机产生一个batch的image和label
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label],
                                                            batch_size=batch_size,
                                                            num_threads=16,  # 填充队列的线程数量
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue=min_queue_examples)

        return images_train, tf.reshape(labels_train, [batch_size])
    else:  # 不对图像进行数据增强处理
        # 1.使用下面这个函数进行剪切操作
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        # 2.剪切完成直接进行标准化
        float_image = tf.image.per_image_standardization(resized_image)

        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch * 0.4)

        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size,
                                                  num_threads=16,
                                                  capacity=min_queue_examples + 3 * batch_size)

        return images_test, tf.reshape(labels_test, [batch_size])
