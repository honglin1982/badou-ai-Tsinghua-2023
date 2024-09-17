from __future__ import print_function, division

from keras.datasets import  mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import  matplotlib.pyplot as plt
import numpy as np
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #设置tf日志等级 只输出错误

import tensorflow as tf
#指定每个进程使用的GPU显存上限
gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpuOptions))

class GAN() :

    def __init__(self):

        self.imgRows = 28
        self.imgCols = 28
        self.channels = 1
        self.imgShape = (self.imgRows, self.imgCols, self.channels)
        self.latentDim = 100  #隐含特征属性向量维度

        optimizer = Adam(0.0002, 0.5)
        # 创建并编译判别器
        self.discriminator = self.buildDiscriminator()
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer,
                                   metrics = ['accuracy'])
        #创建生成器
        self.generator = self.buildGenerator()

        #将生成器 和 判别器(不进行训练的)绑定
        z = Input(shape = (self.latentDim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        validity = self.discriminator(img)
        self.combined = Model(z, validity)
        self.combined.compile(loss = 'binary_crossentropy', optimizer = optimizer)

    def buildDiscriminator(self):

        model = Sequential()
        model.add(Flatten(input_shape = self.imgShape)) #添加扁平层
        model.add(Dense(512))                           #输出512维度
        model.add(LeakyReLU(alpha = 0.2))               #添加LeakyReLU激活函数层
        model.add(Dense(256))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dense(1, activation = 'sigmoid'))
        model.summary()

        img = Input(shape = self.imgShape)
        validity = model(img)
        return Model(img, validity)

    def buildGenerator(self):

        model = Sequential()
        model.add(Dense(256, input_dim = self.latentDim))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.imgShape), activation = 'tanh'))
        model.add(Reshape(self.imgShape))
        model.summary()

        noise = Input(shape = (self.latentDim,))
        img = model(noise)
        return Model(noise, img)

    def train(self, epochs, batchSize = 128, sampleInterval = 50):

        #加载数据集
        (Xtrain, _), (_, _) = mnist.load_data()

        #缩放处理
        Xtrain = Xtrain / 127.5 - 1
        Xtrain = np.expand_dims(Xtrain, axis = 3)

        # batchSize 长度的 真集合 和 假集合容器
        valid = np.ones((batchSize, 1))
        fake = np.zeros((batchSize, 1))
        for epoch in range(epochs):

            #训练判别器
            #随机选择一批次的图片
            idx = np.random.randint(0, Xtrain.shape[0], batchSize)
            imgs = Xtrain[idx]
            noise = np.random.normal(0, 1, (batchSize, self.latentDim))

            #生成器生成一批次图片
            genImgs = self.generator.predict(noise)
            dLossReal = self.discriminator.train_on_batch(imgs, valid)
            dLossFake = self.discriminator.train_on_batch(genImgs, fake)
            dLoss = 0.5 * np.add(dLossReal, dLossFake)

            #训练生成器
            noise = np.random.normal(0, 1, (batchSize, self.latentDim))
            gLoss = self.combined.train_on_batch(noise, valid)
            print("%d [ 判别器 loss : %f, acc : %.2f%%] [生成器 loss : %f]" % (epoch, dLoss[0], 100 * dLoss[1], gLoss))
            #每间隔 sampleInterval 代 保存生成的图片样本
            if epoch % sampleInterval == 0 : self.sampleImages(epoch)

    def sampleImages(self, epoch):

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latentDim))
        genImgs = self.generator.predict(noise)
        genImgs = 0.5 * genImgs + 0.5
        fig, axs = plt.subplots(r, c)
        count = 0;
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(genImgs[count, :, :, 0], cmap = 'gray')
                axs[i, j].axis('off')
                count += 1
        fig.savefig('./images/mnist_%d.png' % epoch)
        plt.close()

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=3000, batchSize = 32, sampleInterval = 200)