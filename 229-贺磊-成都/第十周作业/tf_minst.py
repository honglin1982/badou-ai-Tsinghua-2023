#!/usr/bin/env python
# coding: utf-8



# # 全连接神经网络

# In[113]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/",one_hot=True)


# In[29]:


# 定义输入和输出占位符
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

 


# In[30]:


numClasses = 10
inputSize = 784
numHiddenUnits = 50
trainingIterations = 10000
batchSize = 64


# In[31]:


X = tf.placeholder(tf.float32,shape=[None,inputSize])
Y = tf.placeholder(tf.float32,shape=[None,numClasses])


# In[32]:


# 参数初始化
# 单层神经网络，输入层——》中间层(一层)——》输出层
W1 = tf.Variable(tf.truncated_normal([inputSize,numHiddenUnits], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1),[numHiddenUnits])
W2 = tf.Variable(tf.truncated_normal([numHiddenUnits,numClasses], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1),[numClasses])


# In[33]:


# 网络结构
hiddenLayerOutput = tf.matmul(X,W1) + B1
hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)
finalOutput = tf.matmul(hiddenLayerOutput,W2) + B2
finalOutput = tf.nn.relu(finalOutput)


# In[35]:


# 网络迭代
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = finalOutput)) #损失函数
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(loss) # 优化器


# In[37]:


#准确率测试
correct_prediction = tf.equal(tf.argmax(finalOutput,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))


# In[40]:


# 训练
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(trainingIterations):
    batch = mnist.train.next_batch(batchSize)
    batchInput = batch[0]
    batchLabels = batch[1]
    _,trainingLoss = sess.run([opt,loss],feed_dict={X: batchInput, y_:batchLabels})
    if i%100 == 0:
        trainAccuracy = accuracy.eval(session=sess,feed_dict={X: batchInput, y_:batchLabels})
        print("step %d, train_loss: %g, training accuracy: %g"%(i,trainingLoss,trainAccuracy))


# # 加深网络结构层数

# In[41]:


# 多层神经网络(2层)
numHiddenUnitsLayer2 = 100
trainingIterations = 10000
X = tf.placeholder(tf.float32,shape=([None,inputSize]))
Y = tf.placeholder(tf.float32,shape=([None,numClasses]))

W1 = tf.Variable(tf.random_normal([inputSize,numHiddenUnits],stddev=0.1))
B1 = tf.Variable(tf.constant(0.1),[numHiddenUnits])
W2 = tf.Variable(tf.random_normal([numHiddenUnits,numHiddenUnitsLayer2],stddev=0.1))
B2 = tf.Variable(tf.constant(0.1),[numHiddenUnitsLayer2])
W3 = tf.Variable(tf.random_normal([numHiddenUnitsLayer2,numClasses],stddev=0.1))
B3 = tf.Variable(tf.constant(0.1),[numClasses])


hiddenLayerOutput = tf.matmul(X,W1) + B1
hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)
hiddenLayer2Output = tf.matmul(hiddenLayerOutput,W2) + B2
hiddenLayer2Output = tf.nn.relu(hiddenLayer2Output)
finalOutput = tf.matmul(hiddenLayer2Output,W3) + B3
finalOutput = tf.nn.relu(finalOutput)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=finalOutput))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


correct_prediction = tf.equal(tf.argmax(finalOutput,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

# 训练
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(trainingIterations):
    batch = mnist.train.next_batch(batchSize)
    batchInput = batch[0]
    batchLabels = batch[1]
    _,trainingLoss = sess.run([opt,loss],feed_dict={X: batchInput, y_:batchLabels})
    if i%100 == 0:
        trainAccuracy = accuracy.eval(session=sess,feed_dict={X: batchInput, y_:batchLabels})
        print("step %d, train_loss: %g, training accuracy: %g"%(i,trainingLoss,trainAccuracy))


# # 卷积神经网络

# In[101]:


import tensorflow as tf
import random
import numpy as np
import matplotlib.pylab as plt
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[102]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/",one_hot=True)


# In[103]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
x = tf.placeholder("float",shape=[None,28,28,1]) #输入维度是28*28*1
y_ = tf.placeholder("float",shape=[None,10]) #输出维度


# In[104]:


"""第一大卷积层"""
# 设置卷积核
W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1)) # 5*5的卷积核，1是和前面输入相同的通道数，32是输出通道数。
b_conv1 =  tf.Variable(tf.constant(0.1,shape=[32])) #偏置参数

h_conv1 = tf.nn.conv2d(input=x,filter=W_conv1,strides=[1,1,1,1],padding="SAME") + b_conv1 #卷积层
h_conv1 = tf.nn.relu(h_conv1) #非线性激活函数函数层

h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") #池化，最大池化



# In[105]:


# 用函数产封装
def conv2d(x,W):
    return tf.nn.conv2d(input=x,filter=W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


# In[106]:


# """第二大卷积层"""
W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1)) # 5*5的卷积核，1是和前面输入相同的通道数，32是输出通道数。
b_conv2 = tf.Variable(tf.constant(0.1,shape=[64])) #偏置参数

# h_conv2 = tf.nn.conv2d(h_pool1,filter=W_conv2,strides=[1,1,1,1],padding="SAME") + b_conv2
# h_conv2 = tf.nn.relu(h_conv2)
# 可调用函数
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)


# In[107]:


# 第一个全连接层
W_fc1 = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
b_fc1= tf.Variable(tf.constant(0.1,shape=[1024]))

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)



# In[108]:


# Drop Layer
keep_prob = tf.placeholder("float") #保留率
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)


# In[109]:


# 第二个全连接层
W_fc2 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1,shape=[10]))

# 输出层
y = tf.matmul(h_fc1_drop,W_fc2) + b_fc2 #输出结果


# In[110]:


crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
trainStep = tf.train.AdadeltaOptimizer().minimize(crossEntropyLoss)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))


# In[111]:


sess.run(tf.global_variables_initializer())

batchSize = 64
epochs = 1000
for epoch in range(epochs):
    batch = mnist.train.next_batch(batchSize)
    trainingInputs = batch[0].reshape([batchSize,28,28,1])
    trainingLabels = batch[1]
    _,trainingLoss = sess.run([trainStep,crossEntropyLoss],feed_dict={x: trainingInputs, y_:trainingLabels,keep_prob:0.5})
    if epoch % 100 == 0:
        trainAccuracy = accuracy.eval(session=sess,feed_dict={x: trainingInputs, y_:trainingLabels,keep_prob:0.5})
        print("step %d, training_Loss: %g, training_accuracy: %g"%(epoch,trainingLoss,trainAccuracy))
    trainStep.run(session=sess,feed_dict={x:trainingInputs,y_:trainingLabels,keep_prob:0.5})    
        
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




