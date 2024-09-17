# -*- coding: utf-8 -*-
# @Time    : 2023/8/27 19:53
# @Author  : zgh
# @FileName: GAN.py
# @Software: PyCharm


from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import matplotlib.pyplot as plt

import sys

import numpy as np


class GAN():
	def __init__(self):
		self.img_rows = 28
		self.img_cols = 28
		self.channels = 1
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.latent_dim = 100

		optimizer = Adam(0.0002, 0.5)

		# 构建并且编译分类器
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy',
		                           optimizer=optimizer,
		                           metrics=['accuracy'])

		# 构建并且编译生成器
		self.generator = self.build_generator()
		# 用生成器生成假样本
		z = Input(shape=(self.latent_dim))
		img = self.build_generator(z)
		# 保障判别器不更新参数
		self.discriminator.trainable = False
		# 得到此时判别器给出的标签
		validity = self.discriminator(img)
		# 联合训练 串联生成器和判别器
		self.combined = Model(z,validity)
		self.combined.compile(loss = 'binary_crossentropy', optimizer=optimizer)

	def build_discriminator(self):
		model = Sequential()
		model.add(Flatten(input_dim=self.img_shape))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(256))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(256), activation='sigmod')
		model.summary()
		img = Input(shape=self.img_shape)
		validity = model(img)

		return Model(model, validity)

	def build_generator(self):
		model = Sequential()

		model.add(Dense(256,input_dim=self.latent_dim))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(np.prod(self.img_shape),activation='tanh'))
		model.add(Reshape(self.img_shape))

		model.summary()
		noise = Input(shape=(self.latent_dim))
		img = model(noise)

		return Model(noise, img)

	# 生成器->f1(0) -> 判别器 -> 更新判别器 -> f1(1) -> 更新后的判别器 -> 更新生成器
	# 生成器->f2(0) -> 判别器 -> 更新判别器 -> f2(1) -> 更新后的判别器 -> 更新生成器
	def train(self, epochs, batch_size=128, sample_interval=50):

		# 加载数据
		(X_train, _), (_, _) = mnist.load_data()

		# 归一化
		X_train = X_train / 127.5 - 1.
		X_train = np.expand_dims(X_train, axis=3)

		# Adversarial ground truths
		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		for epoch in range(epochs):

			# ---------------------
			#  训练判别器
			# ---------------------

			# 选择真实图片
			idx = np.random.randint(0, X_train.shape[0], batch_size)
			imgs = X_train[idx]
			# 噪声用于生成虚假图片
			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

			# 生成得到的虚假图片
			gen_imgs = self.generator.predict(noise)

			# 训练判别器 真->valid  假->fake 得到真假损失 更新判别器
			d_loss_real = self.discriminator.train_on_batch(imgs, valid)
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# ---------------------
			#  训练生成器
			# ---------------------
			# 噪声用于生成虚假图片
			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

			# 训练生成器，此时认为生成器产生的白起为valid
			g_loss = self.combined.train_on_batch(noise, valid)

			# Plot the progress
			print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

			# If at save interval => save generated image samples
			if epoch % sample_interval == 0:
				self.sample_images(epoch)

	def sample_images(self, epoch):
		r, c = 5, 5
		noise = np.random.normal(0, 1, (r * c, self.latent_dim))
		gen_imgs = self.generator.predict(noise)

		# Rescale images 0 - 1
		gen_imgs = 0.5 * gen_imgs + 0.5

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
		gan.train(epochs=2000, batch_size=32, sample_interval=200)
