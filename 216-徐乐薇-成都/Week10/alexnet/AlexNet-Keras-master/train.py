from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping  #回调函数,用于在训练过程中保存模型,调整学习率,提前终止训练等
from keras.utils import np_utils                                                            	 #keras工具包,包含了许多有用的函数,比如说np_utils.to_categorical()可以将类别标签转换为one-hot编码
from keras.optimizers import Adam                                                           #优化器,Adam优化器比较常用
from model.AlexNet import AlexNet                                                           #自己写的AlexNet模型
import numpy as np
import utils
import cv2
#from keras import backend as K                                                             #Keras后端,用于设置图像的维度顺序
import keras.backend as K
#K.set_image_dim_ordering('tf')                                                             #设置图像的维度顺序为tf,即(224,224,3).
# keras2.2.5中已经弃用,改为K.set_image_data_format('channels_last')
K.set_image_data_format('channels_last')

#总结一下,所有函数的顺序是：
#1.加载数据
#2.数据预处理,包括归一化,划分训练集和验证集,将标签转换为one-hot编码,生成数据生成器
#3.建立模型
#4.设置回调函数
#5.训练模型
#6.保存模型
#7.测试模型

def generate_arrays_from_file(lines,batch_size):                           #数据生成器,节省内存,逐步提升模型的训练,这里的lines是训练集的txt,lines的每一行都是图片的路径和标签,比如说"image/1.jpg;0"
    # 获取总长度
    n = len(lines)                                                                          #获取训练集的长度
    i = 0                                                                                   #i用于记录当前训练的位置
    while 1:
        X_train = []                                                                        #X_train用于存储图片
        Y_train = []                                                                        #Y_train用于存储标签
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)                                                    #每次开始训练前都打乱一下训练集
            name = lines[i].split(';')[0]                                                   #获取图片的路径
            # 从文件中读取图像
            img = cv2.imread(r".\data\image\train" + '/' + name)                            #读取图片,注意,是相对于dataset_process.py的相对路径。
            # r是防止字符转义,比如说\n,\t等.+'/'是为了在路径中加入/,因为lines[i].split(';')[0]只是图片的名字,没有路径，不加'\'是因为在Windows下,路径是用'\'分割的,而'\'在Python中是转义字符,所以要用'/'代替
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img/255                                                                   #归一化，这样做可以提升模型的训练精度
            X_train.append(img)                                                             #将图片加入X_train
            Y_train.append(lines[i].split(';')[1])                                          #将标签加入Y_train
            # 读完一个周期后重新开始
            i = (i+1) % n                                                                   #如果i+1到达n,则从0开始,否则继续增加
        # 处理图像
        X_train = utils.resize_image(X_train,(224,224))                                     #将图片缩放到224*224
        X_train = X_train.reshape(-1,224,224,3)                                             #将图片的维度变为(224,224,3)，其中-1表示不确定,由计算机自己计算
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= 2)                 #将标签转换为one-hot编码，比如说0转换为[1,0],1转换为[0,1]。np.array(Y_train)将Y_train转换为numpy数组,这样才能使用np_utils.to_categorical()函数
        yield (X_train, Y_train)                                                            #返回数据和标签


if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r".\data\dataset.txt","r") as f:                                              #打开数据集的txt
        lines = f.readlines()                                                                         #读取数据集的txt

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)                                                                   #设置随机数种子,保证每次生成的随机数相同
    np.random.shuffle(lines)                                                                  #打乱数据
    np.random.seed(None)                                                                    #恢复随机数种子，这样做是为了防止后面的随机数受到影响

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)                                                           	#获取训练集的长度
    num_train = len(lines) - num_val                                                        #获取验证集的长度

    # 建立AlexNet模型
    model = AlexNet()
    
    # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(                                                               #ModelCheckpoint是一个回调函数,属于keras.callbacks模块,在每个世代之后保存模型
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', #保存的路径,其中epoch是世代数,loss是训练集的loss,val_loss是验证集的loss,epoch:03d表示epoch用3位表示,不足3位的用0补齐,loss:.3f表示loss用3位小数表示,不足3位的用0补齐,val_loss:.3f同理
                                    monitor='acc',                                                      #监视的数据,这里是监视验证集的准确率
                                    save_weights_only=False,                                            #是否只保存权重,这里是保存整个模型
                                    save_best_only=True,                                                #是否只保存最好的模型,一般保存最好的模型可以防止模型过拟合
                                    period=3                                                            #保存的频率,这里是每3个世代保存一次
                                )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(                                                                      #ReduceLROnPlateau是一个回调函数,属于keras.callbacks模块,当评价指标不再提升时，减少学习率
                            monitor='acc',                                                              #监视的数据,这里是监视验证集的准确率
                            factor=0.5,                                                                 #学习率每次下降为原来的一半
                            patience=3,                                                                 #acc三次不下降就下降学习率
                            verbose=1                                                                   #输出学习率信息
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(                                                                     #EarlyStopping是一个回调函数,属于keras.callbacks模块,当监测值不再改善时，该回调函数将中止训练
                            monitor='val_loss',                                                         #监视的数据,这里是监视验证集的loss
                            min_delta=0,                                                                #被认为是提升的最小变化量,即变化量小于min_delta,则认为没有提升
                            patience=10,                                                                #val_loss不下降的世代数,即10个世代val_loss没有下降,则停止训练
                            verbose=1                                                                   #输出EarlyStopping信息
                        )

    # 交叉熵
    model.compile(loss = 'categorical_crossentropy',                                           #交叉熵损失函数,用于多分类问题
            optimizer = Adam(lr=1e-3),                                                                  #Adam优化器,学习率为0.001
            metrics = ['accuracy'])                                                                     #评价函数为准确率

    # 一次的训练集大小
    batch_size = 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size)) #打印训练集的大小,验证集的大小,一次的训练集大小
    
    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),                        #fit_generator是一个函数,属于keras.models模块,用于使用Python生成器或Sequence实例逐批生成的数据上训练模型
            steps_per_epoch=max(1, num_train//batch_size),                                               #每个世代需要训练的次数,即训练集的大小除以一次的训练集大小
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),                    #验证集的生成器
            validation_steps=max(1, num_val//batch_size),                                                #每个世代需要验证的次数,即验证集的大小除以一次的训练集大小
            epochs=50,                                                                                   	  #训练的世代数
            initial_epoch=0,                                                                            	  #初始世代数
            callbacks=[checkpoint_period1, reduce_lr])                                                     #回调函数
    model.save_weights(log_dir+'last1.h5')                                                                   #保存模型权重,保存在logs文件夹下

