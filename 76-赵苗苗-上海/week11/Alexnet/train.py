#查看日志，导入权重的回调函数（checkpoint_period1）、学习率下降的回调函数（reduce_lr）和早停的回调函数（early_stopping）
from keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras.utils import np_utils  #将标签数据转换为独热编码形式
from keras.optimizers import adam  #导入Adam优化器，用于模型训练
from model.AlexNet import Alexnet
import numpy as np
import utils
import cv2
from keras import backend as K

#generate_arrays_from_file函数用于从文件中读取图像数据，进行预处理和标签处理，返回返回生成器对象
def generate_arrays_from_file(lines,batch_size):
    #获取总样本数n
    n=len(lines)
    i=0
    #创建一个无限循环的迭代过程，用于不断生成训练数据的批次
    while 1:
        x_train=[]
        y_train=[]
        #获得一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)  #随机打乱
            name=lines[i].split(',')[0]   #从样本列表 ​lines​中获取文件名
            #从文件中读取图像
            img=cv2.imread('D:\data\image\train'+'\\'+name)
            if img is None:
                print(f'Failed to load image:{name}')
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=img/255  #对图像归一化
            x_train.append(img)  #将图像数据添加到x_train列表
            y_train.append(lines[i].split(',')[1])  #将对应标签添加到y_train列表
            #读完一个周期后重新开始
            i=(i+1)%n
            #处理图像
            x_train=utils.resize_image(x_train,(224,224)) #调整尺寸
            x_train=x_train.reshape(-1,224,224,3)  #调整形状
            y_train=np_utils.to_categorical(np.array(y_train),num_classes=2)  #设置类别数为2
            yield(x_train,y_train)  #生成一个训练批次的输入数据和标签，在训练模型的时候会作为参数传递给fit_generator方法，用于动态生成训练数据

#定义主函数main
if __name__=='__main__':
    #模型保存的位置
    log_dir='./logs/'
    #打开数据集的txt
    with open(r"D:\data\dataset.txt",'r') as f:
        lines=f.readlines()

    #打乱行，这个txt主要用来帮助读取数据来训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    #90%用于训练，10%用于估计
    num_val=int(len(lines)*0.1)
    num_train=len(lines)-num_val

    #建立AlexNet模型
    model=Alexnet()

    #回调函数ModelCheckpoint()，在训练过程中定期保存模型的权重文件
    #根据指定的格式生成文件名，并检测准确率为指标
    checkpoint_period1=ModelCheckpoint(
        log_dir+'ep{epc:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',  #指定保存模型权重文件路径和文件名的格式
        monitor='acc',  #指定模型指标检测的参数，acc准确率
        save_best_only=True, 
        save_weights_only=False,
        period=3  #指定保存模型的间隔周期
    )

    #回调函数ReduceLROnPlateau()，在训练过程中根据检测的指标来动态的调整学习率
    reduce_lr=ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1    #当模型的准确率在3个epoch内都没有提升时，学习率会缩小为原来的一半
    )
    #回调函数EarlyStopping()，用于在训练过程中监测验证集损失，并在满足特定条件时停止训练
    early_stopping=EarlyStopping(
        monitor='val_loss',  #指定模型指标监测的参数，val_loss验证集的损失
        min_delta=0,
        patience=10,
        verbose=1    #在每个epoch结束后，检查验证集损失是否改善，如果连续10个epoch都没有达到预期，停止训练
    )
    #配置模型的训练参数，使用交叉熵作为损失函数，使用准确率作为评估指标来检测模型的性能
    model.compile(
        loss='categorical_crossentropy',
        optimizer=adam(lr=1e-3),  
        metrics=['accuracy']
    )
    #一次训练集的大小
    batch_size=128
    print('Train on {} samples,val on {} samples,with batch size{}.'.format(num_train,num_val,batch_size))

    #开始训练
    #使用fit_generator()函数训练模型（使用生成器产生数据的模型的方法）
    model.fit_generator(
        generate_arrays_from_file(lines[:num_train],batch_size),  #指定一个生成器函数，用于产生训练数据
        steps_per_epoch=max(1,num_train//batch_size),  #指定了训练的步数epochs,每个epoch中要执行的训练步骤的数量
        validation_data=generate_arrays_from_file(lines[num_train:],batch_size),  #指定了一个生成器函数，用于产生训练数据
        validation_steps=max(1,num_val//batch_size),   #指定了验证的步数，每个epoch中要执行的验证步骤的数量
        epochs=50,   #指定了训练的总轮步数
        initial_epoch=0,  #指定初始轮数
        callbacks=[checkpoint_period1,reduce_lr]  #指定了一些回调函数
    )
    #保存模型的权重到文件中
    model.save_weights(log_dir+'last1.h5')
