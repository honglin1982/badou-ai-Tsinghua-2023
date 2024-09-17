#该文件的目的是构造神经网络的整体结构，并进行训练和测试（评估）过程
import tensorflow as tf
import numpy as np
import time #用于计算运行时间
import math #用于计算精度
import Cifar10_data #用于读取Cifar10数据集

max_steps=4000 #最大迭代次数
batch_size=100 #每次迭代的样本数
num_examples_for_eval=10000 #测试集样本数
data_dir="Cifar_data/cifar-10-batches-bin" #Cifar10数据集的路径

#创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
def variable_with_weight_loss(shape,stddev,w1): #shape表示生成张量的维度，stddev表示标准差，w1表示L2 loss的大小
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev)) #生成一个截断的正态分布,标准差为stddev,形状为shape的张量
    if w1 is not None: #如果w1不为空
        weights_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weights_loss") #计算权重L2 loss与w1的乘积
        tf.add_to_collection("losses",weights_loss) #将最终的结果放在名为losses的集合里面
    return var #返回生成的变量

#使用上一个文件里面已经定义好的文件序列读取函数读取训练数据文件和测试数据从文件.
#其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
images_train,labels_train=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True) #读取训练数据文件
images_test,labels_test=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)   #读取测试数据文件

#创建x和y_两个placeholder，用于在训练或评估时提供输入的数据和对应的标签值。
#要注意的是，由于以后定义全连接网络的时候用到了batch_size，所以x中，第一个参数不应该是None，而应该是batch_size
x=tf.placeholder(tf.float32,[batch_size,24,24,3]) #第一个参数是batch_size,表示一次喂入神经网络的样本数,第二个参数是图片的尺寸,第三个参数是图片的通道数
y=tf.placeholder(tf.int32,[batch_size])           #标签值

#创建第一个卷积层 shape=(kh,kw,ci,co)
kernel1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0) #使用上面定义好的函数创建第一个卷积层的权重变量
conv1=tf.nn.conv2d(x,kernel1,[1,1,1,1],padding="SAME")                 #使用tf.nn.conv2d()函数对输入的x和权重kernel1进行卷积操作
bias1=tf.Variable(tf.constant(0.0,shape=[64]))                         #创建偏置项,并初始化为0,shape=(64,)
relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))                          #使用tf.nn.bias_add()函数将偏置项bias1加到卷积结果上，并使用tf.nn.relu()函数进行激活操作
pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME") #relu1经过最大池化操作,池化窗口大小为3*3,步长为2*2,池化后的结果为pool1,shape=(batch_size,12,12,64)

#创建第二个卷积层
kernel2=variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0) #使用上面定义好的函数创建第二个卷积层的权重变量
conv2=tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")              #使用tf.nn.conv2d()函数对输入的pool1和权重kernel2进行卷积操作
bias2=tf.Variable(tf.constant(0.1,shape=[64]))                          #创建偏置项,并初始化为0.1,shape=(64,)
relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias2))                           #使用tf.nn.bias_add()函数将偏置项bias2加到卷积结果上，并使用tf.nn.relu()函数进行激活操作
pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME") #使用tf.nn.max_pool()函数进行最大池化操作

#因为要进行全连接层的操作，所以这里使用tf.reshape()函数将pool2输出变成一维向量，并使用get_shape()函数获取扁平化之后的长度
reshape=tf.reshape(pool2,[batch_size,-1])    #这里面的-1代表将pool2的三维结构拉直为一维结构
dim=reshape.get_shape()[1].value             #get_shape()[1].value表示获取reshape之后的第二个维度的值

#建立第一个全连接层
weight1=variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004) #使用上面定义好的函数创建第一个全连接层的权重变量
fc_bias1=tf.Variable(tf.constant(0.1,shape=[384]))                      #创建偏置项,并初始化为0.1,shape=(384,)
fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)                    #使用tf.matmul()函数进行（矩阵相乘操作 + 偏置项 ）* 激活函数

#建立第二个全连接层
weight2=variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004) #使用上面定义好的函数创建第二个全连接层的权重变量
fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))                      #创建偏置项,并初始化为0.1,shape=(192,)
local4=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)                     #使用tf.matmul()函数进行（矩阵相乘操作 + 偏置项 ）* 激活函数

#建立第三个全连接层
weight3=variable_with_weight_loss(shape=[192,10],stddev=1 / 192.0,w1=0.0)   #使用上面定义好的函数创建第三个全连接层的权重变量
fc_bias3=tf.Variable(tf.constant(0.1,shape=[10]))                           #创建偏置项,并初始化为0.1,shape=(10,)
result=tf.add(tf.matmul(local4,weight3),fc_bias3)                           #使用tf.matmul()函数进行（矩阵相乘操作 + 偏置项 ）

#计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y,tf.int64)) #使用tf.nn.sparse_softmax_cross_entropy_with_logits()函数计算交叉熵损失
                                                                        # 这个函数的第一个参数是神经网络不包括softmax层的前向传播结果，第二个参数是训练数据的正确答案

weights_with_l2_loss=tf.add_n(tf.get_collection("losses"))              #使用tf.add_n()函数将get_collection()函数返回的元素加起来，得到最终的正则化损失
loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss                 #reduce_mean()函数用于计算交叉熵平均值，最终得到的损失函数为loss.将交叉熵损失和正则化损失相加，得到最终的损失函数

train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)                    #使用tf.train.AdamOptimizer()函数优化损失函数，得到训练操作train_op

#函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
top_k_op=tf.nn.in_top_k(result,y,1)                                    #使用tf.nn.in_top_k()函数计算输出结果中top 1的准确率

init_op=tf.global_variables_initializer()                               #初始化所有的变量,并将初始化后的变量赋值给init_op
with tf.Session() as sess:                                              #创建一个会话，并通过Python中的上下文管理器来管理这个会话
    sess.run(init_op)                                                   #初始化变量
    #启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()                                      #启动线程操作

#每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range (max_steps):                                      #在训练过程中不断的调用train_op来优化神经网络的参数
        start_time=time.time()                                          #计算每一个batch的处理时间
        image_batch,label_batch=sess.run([images_train,labels_train])   #通过sess.run()函数来获取一个batch的训练数据
        _,loss_value=sess.run([train_op,loss],feed_dict={x:image_batch,y:label_batch}) #通过sess.run()函数来运行训练操作train_op和损失函数的计算操作loss.其中的参数feed_dict用于指定输入的训练数据
        duration=time.time() - start_time                               #计算每一个batch的处理时间

        if step % 100 == 0:                                             #每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
            examples_per_sec=batch_size / duration                      #计算每秒钟能训练的样本数量
            sec_per_batch=float(duration)                               #计算训练一个batch数据所花费的时间
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))

#计算最终的正确率
    num_batch=int(math.ceil(num_examples_for_eval/batch_size))  #math.ceil()函数用于求整,计算总共需要多少轮迭代来完成测试集上的测试,并通过num_batch来记录迭代的轮数
    true_count=0                                                #定义变量true_count来记录预测正确的样例个数
    total_sample_count=num_batch * batch_size                   #计算总共需要测试的样例个数

    #在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):                                      #在一个for循环里面统计所有预测正确的样例个数
        image_batch,label_batch=sess.run([images_test,labels_test]) #通过sess.run()函数来获取一个batch的测试数据
        predictions=sess.run([top_k_op],feed_dict={x:image_batch,y:label_batch}) #通过sess.run()函数来运行top_k_op操作，得到在当前batch中预测正确的样例个数
        true_count += np.sum(predictions)                           #将当前batch中预测正确的样例个数累加到true_count变量中

    #打印正确率信息
    print("accuracy = %.3f%%"%((true_count/total_sample_count) * 100)) #计算正确率，将预测正确的样例个数除以总共需要预测的样例个数，得到正确率
