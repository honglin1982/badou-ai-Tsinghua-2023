#使用TensorFlow Keras构建神经网络模型来识别手写数字的项目

#1、加载训练数据和测试数据，并打印出它们的形状和标签
#获取mnist手写数字识别数据集
#train_images——训练集，test_images——测试集，train_labels——训练集标签，test_labels——测试集标签
from tensorflow.keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

#2、取出测试数据中的第一张图片，并使用matplotlib库将其显示出来
import matplotlib.pyplot as plt
digit=test_images[0]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()

#3、构建一个神经网络模型，其中包括两个全连接层
#完成神经网络模型的搭建，配置和编译工作
from tensorflow.keras import models
from tensorflow.keras import layers
#创建一个序贯模型，是一系列层的线性堆叠
network=models.Sequential()  
#向模型中添加两个全连接层
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax')) 
#对模型进行编译设置
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',
                metrics=['accuracy'])

#4、对训练数据和测试数据进行预处理，将二维数组转换为一维数组，并进行归一化处理
#独热编码：to_categorical​ 函数默认将标签数组中的最小值作为类别的起始索引，从0开始
from tensorflow.keras.utils import to_categorical
train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255

test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float')/255

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

#5、使用训练数据对神经网络模型进行训练
network.fit(train_images,train_labels,epochs=5,batch_size=128)

#6、使用测试数据对训练好的神经网络模型进行评估，并打印出测试损失和准确率
test_loss,test_acc=network.evaluate(test_images,test_labels,verbose=1)
print('test_loss',test_loss)
print('test_acc',test_acc)
#7、从测试数据中取出第二张图片，并使用网络模型进行预测，输出识别结果
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
digit=test_images[1]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
#使用网络模型预测哪个图片是1
test_images=test_images.reshape((10000,28*28))
res=network.predict(test_images)
for i in range(res[1].shape[0]):
    if(res[1][i]==1):
        print('the number for the picture is :',i)
        break