import tensorflow as tf
import numpy as np
import time
import math
import Cifar10Data

maxSteps = 4000
batchSize = 100
numExamplesForEval=10000
dataDir = "Cifar_data/cifar-10-batches-bin"

#创建函数记录权重L2 weightLoss
def variableWithWeightLoss(shape, stddev, w1):

    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev)) #生成shape大小的标准差为stddev的符合正态分布的随机数
    if w1 is not None:
        weightLoss = tf.multiply(tf.nn.l2_loss(var), w1, name='weightLoss')
        tf.add_to_collection("losses", weightLoss)
    return var

#利用Cifar10Data读取训练和测试数据
imagesTrain, labelsTrain = Cifar10Data.inputs(dataDir=dataDir, batchSize=batchSize, distorted=True)
imagesTest, labelsTest = Cifar10Data.inputs(dataDir=dataDir, batchSize=batchSize, distorted=None)

#创建两个占位 由于全连接网络中用到了batchSize，所以第一个参数不能为None
x = tf.placeholder(tf.float32, [batchSize, 24, 24, 3])
y = tf.placeholder(tf.int32, [batchSize])

#第一层卷积 shape = (kh, kw, ci, co)
kernel1 = variableWithWeightLoss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")# ksize [n(数量)、 h、 w、 c]

#第二层卷积
kernel2 = variableWithWeightLoss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

#进行全连接层操作前 将数据压缩为一维
oneDim = tf.reshape(pool2, [batchSize, -1])
dim = oneDim.get_shape()[1].value  #获取压缩后的数据个数

#第一层全连接
weight1 = variableWithWeightLoss(shape=[dim, 384], stddev=0.04, w1=0.004)
fcBias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc1 = tf.nn.relu(tf.matmul(oneDim, weight1) + fcBias1)

#第二层全连接
weight2 = variableWithWeightLoss(shape=[384, 192], stddev=0.04, w1=0.004)
fcBias2 = tf.Variable(tf.constant(0.1, shape=[192]))
local = tf.nn.relu(tf.matmul(fc1, weight2) + fcBias2)

#第三层全连接
weight3 = variableWithWeightLoss(shape=[192,10], stddev= 1 / 192.0, w1=0.0)
fcBias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.nn.relu(tf.matmul(local, weight3) + fcBias3)

#计算损失 权重参数正则化损失和交叉熵损失
crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,
               labels=tf.cast(y, tf.int64))
weightWithL2Loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(crossEntropy) + weightWithL2Loss
trainOp = tf.train.AdamOptimizer(1e-3).minimize(loss)

#获取topk 的准确率, 函数默认k值为1
topKOp = tf.nn.in_top_k(result, y, 1)
initOp = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(initOp)
    tf.train.start_queue_runners() #启动线程操作加速
    for step in range(maxSteps):
        startTime = time.time()
        imageBatch, labelBatch = sess.run([imagesTrain, labelsTrain])
        _, lossValue = sess.run([trainOp, loss], feed_dict={x : imageBatch, y : labelBatch})
        duration = time.time() - startTime

        if step % 100 == 0 :
            examplesPerSec = batchSize / duration
            secPerBatch = float(duration)
            print("step %d, loss=%.2f(%.1f examples/sec %.3f sec/batch"%
                  (step, lossValue, examplesPerSec, secPerBatch))

#计算准确率
    numBatch = int(math.ceil(numExamplesForEval/batchSize))
    trueCount = 0;
    total = numBatch * batchSize
    for i in range(numBatch):
        imageBatch, labelBatch = sess.run([imagesTest, labelsTest])
        predictions = sess.run([topKOp], feed_dict={x:imageBatch, y:labelBatch})
        trueCount += np.sum(predictions)
    print("accuracy = %.3f%%"%(trueCount / total * 100))