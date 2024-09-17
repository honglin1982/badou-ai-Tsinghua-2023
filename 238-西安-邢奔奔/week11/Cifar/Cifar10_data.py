import os
import tensorflow as tf

num_classes = 10
# 设定用于训练和评估的样本总数
num_exampes_pre_epoch_for_train = 50000
num_exampes_pre_epoch_for_eval = 10000


# 定义一个空类来接返回值
class CIFAR10Record(object):
    pass


# 定义一个读取数据的函数，来读取目标文件的内容
def read_cifar10(file_queue):
    # 实例化一个类，并且设置长宽深，
    result = CIFAR10Record()

    label_bytes = 1
    result.high = 32
    result.wide = 32
    result.depth = 3
    # 计算每个图像的元素数量
    image_bytes = result.high * result.wide * result.depth
    # 计算每个样本包含的元素数量
    record_bytes = image_bytes + label_bytes
    # 使用tf.FixedLengthRecordReader来创建一个文件读取类，该类的目的就是读取文件
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # 该类的read()函数从文件队列中读取文件
    result.key, value = reader.read(file_queue)
    # 通过decode解码，来实现文件读取后从字符串模式解析到图像对应的像素
    record_bytes = tf.decode_raw(value, tf.uint8)
    # 使用tf.strided_slice(record_bytes,[0],[label_bytes])将标签提取出来 并将其改变为tf.uint32格式
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.uint32)
    # 提取图像数，将一维数据转换为三维数据，切片得到的子张量重新调整为具有给定尺寸 [result.depth, result.high, result.wide] 的张量。
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.high, result.wide])
    # 将分割好的图片数据转换为高宽深的顺序，即变为（hwc）
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    # 返回读出的数据
    return result


def inputs(data_dir, batch_size, distorted):
    '''
    通过这个函数实现数据的预处理
    :param data_dir:
    :param batch_size:
    :param distorted: 不为None则需要进行预处理
    :return:
    '''
    # 拼接地址
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
    # 用于创建一个包含文件名的输入队列的函数
    file_queue = tf.train.string_input_producer(filenames)
    # 通过自定义的read函数来将队列中的数读出来
    read_input = read_cifar10(file_queue)
    # 通过cast函数来将图像转换为tf.float32格式
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    num_exampes_pre_epoch = num_exampes_pre_epoch_for_train
    # 这里不为None，则为需要对图片进行处理
    if distorted != None:
        # 首先对图像进行随机裁剪，将图像形状变为【24，24，3】的格式
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
        # 将裁剪处理过的图像进行随机左右翻转
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        # 将翻转处理后的图像进行随机亮度调整
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)
        # 将亮度调整处理后的图像进行对比度调节
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=0.8)
        # 将图像进行归一化处理，此处tf.image.per_image_standardization(adjusted_contrast)是将每一个像素减去平均值并除以像素方差
        float_image = tf.image.per_image_standardization(adjusted_contrast)
        # 设置图像数据及标签，因为上面是随机进行裁剪的，所以这里还要进行一次设置
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        # 生成一个最小的数据队列大小，这么做通常是为了确保输入管道中有足够的数据用于进行评估
        min_queue_examples = int(num_exampes_pre_epoch_for_eval * 0.4)

        print('Filling queue with %d CIFAR images before starting to train. This will take a few minutes.'
              % min_queue_examples)
        # 从输入队列随机生成大小为batch_size的数据，
        image_trains, labels_trains = tf.train.shuffle_batch([float_image, read_input.label], batch_size
        =batch_size, num_threads=16,
                                                             capacity=min_queue_examples + 3 * batch_size,
                                                             min_after_dequeue=min_queue_examples, )
        return image_trains, tf.reshape(labels_trains, [batch_size])
    else:
        # 可以理解为只要是进入else就是处理预测数据，对图像进行裁剪或填充，具体看后面的高和宽参数
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)

        float_image = tf.image.per_image_standardization(resized_image)
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_exampes_pre_epoch * 0.4)
        image_test, label_test = tf.train.batch([float_image, read_input.label],
                                                batch_size=batch_size,
                                                num_threads=26, capacity=min_queue_examples + 3 * batch_size)
        return image_test, tf.reshape(label_test, [batch_size])
