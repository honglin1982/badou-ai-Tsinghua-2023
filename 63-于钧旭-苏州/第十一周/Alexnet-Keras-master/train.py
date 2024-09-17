from keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import numpy as np
from keras.optimizers import Adam
from keras.utils import np_utils
import utils
from model import AlexNet
import cv2
from keras import backend as K
K.set_image_data_format('channels_last') #nhwc

def generate_arrays_from_file(lines,batch_size):
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(r".\data\image\train"+'/'+name)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img/255
            X_train.append(img)
            Y_train.append(lines.split(';')[1])
            #读完一个周期后重新开始
            i = (i+1)%n
        # 处理图像
        X_train = utils.resize_image(X_train,(224,224))
        X_train = X_train.reshape(-1,224,224,3)
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes = 2)
        yield (X_train,Y_train)

if __name__ =="__main__":
    #模型保存的位置
    log_dir = "./logs/"
    #打开数据集的txt
    with open(r".\data\dataset.txt","r") as f:
        lines = f.readlines()
    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    #建立模型
    model = AlexNet()
    # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )
    #交叉熵
    model.compile(
        loss='categorical_crossentropy',
        optmizer = Adam(lr=1e-3),
        metrics = ['accuracy']
    )
    batch_size = 128
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    #开始训练
    model.fit_generator(
        generate_arrays_from_file(lines[:num_train],batch_size), # 用于生成训练数据的数据生成器
        steps_per_epoch=max(1,num_train//batch_size), # 每个训练 epoch 中的迭代步数
        validation_data=generate_arrays_from_file(lines[num_train:], batch_size), # 用于生成验证数据的数据生成器
        validation_steps=max(1,num_val//batch_size), # 每个验证 epoch 中的迭代步数
        epochs=50, # 总的训练轮数
        initial_epoch=0, # 开始的训练轮数
        callbacks=[checkpoint_period1,reduce_lr] # 用于回调的列表，例如模型检查点和学习率衰减
    )
    model.save_weights(log_dir+'last1.h5')
