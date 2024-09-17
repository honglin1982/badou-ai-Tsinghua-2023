import os
import tensorflow as tf

#分类数
numClasses = 10
#训练和评估样本数
numExamplesPreEpochForTrain = 50000
numExamplesPreEpochForEval = 10000

class CIFAR10Record(object):
        pass

#读取CIFAR10数据的函数
def readCifar10(fileQueue):

    result = CIFAR10Record()
    labelBytes = 1          #对Cifar10数据集的标记1
    result.height = 32      #高 * 宽 * 通道数
    result.width =  32
    result.depth =  3

    imageBytes = result.height * result.width * result.depth  #图片占用字节数
    recordBytes = labelBytes + imageBytes                     #数据字节数 标记 + 图片字节数

    reader = tf.FixedLengthRecordReader(record_bytes=recordBytes) #创建一个文件读取类
    result.key, value = reader.read(fileQueue)                    #利用read()函数读取文件
    recordBytes = tf.decode_raw(value, tf.uint8)                  #读入记录中

    #利用strided_slice()函数将标签提取出来，并使用tf.cast进行数据类型转换
    result.label = tf.cast(tf.strided_slice(recordBytes, [0], [labelBytes]), tf.int32)

    majorData = tf.reshape(tf.strided_slice(recordBytes, [labelBytes], [imageBytes + labelBytes]),
                           [result.depth, result.height, result.width])

    result.uint8image = tf.transpose(majorData, [1, 2, 0]) #将 chw 变成 hwc
    return result

def inputs(dataDir, batchSize, distorted):

    #构建文件地址列表
    filenames = [os.path.join(dataDir, "data_batch_%d.bin"%i) for i in range(1, 6)]
    #将列表转换成字符串输入队列
    fileQueue = tf.train.string_input_producer(filenames)
    readInput = readCifar10(fileQueue)
    reshapedImage = tf.cast(readInput.uint8image, tf.float32) #转换数据类型为浮点型

    numExamplesPerEpoch = numExamplesPreEpochForTrain #用于设定集合的最小批次
    if distorted != None: #根据 distorted参数是否为空，来判定是否对图片进行增强操作
        croppedImage = tf.random_crop(reshapedImage, [24, 24, 3])                               #首先对图片进行随机裁剪
        flippedImage = tf.image.random_flip_left_right(croppedImage)                            #再对图像进行随机左右翻转
        adjustedBrightness = tf.image.random_brightness(flippedImage, max_delta=0.8)            #再对图像进行随机亮度调整
        adjuestedContrast = tf.image.random_contrast(adjustedBrightness, lower=0.2, upper=1.8)  #再对图像进行随机对比度调整

        floatImage = tf.image.per_image_standardization(adjuestedContrast) #对图像进行标准化处理 对每个元素减去平均值再除以方差
        minQueueExamples = int(numExamplesPreEpochForEval * 0.4)
        floatImage.set_shape([24, 24, 3])                                  #重新设定图片数据与标签数据大小
        readInput.label.set_shape([1])

        print("Inputing queue with %d CIFAR images before starting to train. It will take a few minutes."%minQueueExamples)
        imageTrain, labelTrain = tf.train.shuffle_batch([floatImage, readInput.label], batch_size=batchSize,
                                                        num_threads=16, capacity= minQueueExamples + 3 * batchSize,
                                                        min_after_dequeue=minQueueExamples)
        return imageTrain, tf.reshape(labelTrain, [batchSize])
    else:
        #读入测试集
        resizedImage = tf.image.resize_image_with_crop_or_pad(reshapedImage, 24, 24)
        floatImage = tf.image.per_image_standardization(resizedImage)
        floatImage.set_shape([24, 24, 3])
        readInput.label.set_shape([1])

        minQueueExamples = int(numExamplesPerEpoch * 0.4)
        imageTest, labelTest = tf.train.batch([floatImage, readInput.label], batch_size=batchSize,
                                              num_threads=16, capacity=minQueueExamples + 3 * batchSize)
        return imageTest, tf.reshape(labelTest, [batchSize])