from keras.datasets import mnist
from keras.layers import Dense, BatchNormalization, Flatten, Reshape
from keras.layers import Input, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
# ----------------------------------------------------------------------------- #
#  gen1 -> p1 -> disc1 -> 1(0) -> loss 0-1 -> disc2 + p1 -> 0(1) -> loss 1-0
#  -> gen2 -> p2 -> disc2 -> 1(0) -> loss -> disc3 + p2 -> 0(1) -> loss -> gen3
# ----------------------------------------------------------------------------- #

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


class GAN:
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.img_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        # 随机输入的向量的长度
        self.latent_dim = 100

        # optimizer
        optimizer = Adam(0.0002, 0.5)

        # 创建、编译判别器
        self.discriminator = self.build_discriminator()
        # 判别器单独训练
        self.discriminator.compile(loss="binary_crossentropy",
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # 生成器训练需要用到判别器，他是一个组合网络，但是生成器训练的时候判别器不训练
        # 创建生成器
        self.generator = self.build_generator()

        x = Input(shape=(self.latent_dim,))
        img = self.generator(x)

        # 由于不参与训练，将判别器参数设置为False
        self.discriminator.trainable = False

        # 组合、编译模型
        validity = self.discriminator(img)
        self.combined = Model(x, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # 加载手写数字数据集
        (x_train, _), (_, _) = mnist.load_data()
        # 归一化像素到-1，1
        x_train = x_train / 127.5 - 1
        # 更改shape [N, h, w] -> [N, h, w, 1]
        x_train = np.expand_dims(x_train, axis=3)

        # 标签
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # 在数据集中寻找一组随机批次的图片
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 根据上面的随机数组通过生成器生成一组随机图片
            gen_imgs = self.generator.predict(noise)

            # 计算判别器的损失
            # 判别器处理真实数据时标签为1，希望判别器最小化真实数据的loss，使输出接近于1
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            # 判别器处理生成数据时标签为0，希望判别器最小化生成数据的loss，使输出接近于0（目标为判断为0）
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # 对于生成的假样本来说，我们希望以假乱真，于是我们将标签设置为1，由于判别器没有参与训练
            # 他会正常判断为0，以此来训练我们的生成器，最小化其loss，以接近于1
            g_loss = self.combined.train_on_batch(noise, valid)

            # 打印进度
            print("%d [D loss: %f, acc.: %.2f%%]  [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # 归一化到0-1
        gen_imgs = gen_imgs * 0.5 + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig("./images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(10000, 32, 1000)
