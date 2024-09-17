# _*_ coding : utf-8 _*_
# @Time : 2023/8/22 14:51
# @Author : weixing
# @FileName : gan
# @Project : cv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
import pandas

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# batch_size=64 图片大小：1,28,28
# 定义生成器
# 输入是长度为100的噪声（正态分布随机数）
# 输出为（1,28,28）的图片

# linear1:100---256
# linear2:256---512
# linear1:512---28*28
# linear2:28*28---(1,28,28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):  # x表示长度为100的噪声输入
        img = self.main(x)
        img = img.view(-1, 28, 28)
        return img


# 定义判别器
# 输入为（1,28,28）的图片，输出为二分类的概率值，输出使用sigmoid激活
# BCELose计算交叉熵损失
# nn.LeakyReLU  f(x):x>0,输出x，如果x<0,输出a*x，a表示一个很小的斜率，比如0.1
# 判别器中使用nn.LeakyReLU
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.main(x)
        return x


def gen_img_plot(model, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    # detach()截断梯度,np.squeeze可以去掉维度为1
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):  # prediction.size(0)=16
        plt.subplot(4, 4, i + 1)
        plt.imshow((prediction[i] + 1) / 2)  # tanh 得到的是-1 - 1之间，-》0-1之间
        plt.axis("off")
    plt.show()


if not os.path.exists("model"):
    os.makedirs("model")

# 对数据做归一化（-1 --- 1）
tranform = transforms.Compose([
    transforms.ToTensor(),  # 0-1归一化：channel,high,width
    transforms.Normalize(0.5, 0.5),  # 均值，方差均为0.5
])

train_ds = torchvision.datasets.MNIST("data", train=True, transform=tranform, download=True)

dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

imgs, _ = next(iter(dataloader))
# print(imgs.shape)  # torch.Size([64, 1, 28, 28])

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda' if 0 else 'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)
d_optim = torch.optim.Adam(dis.parameters(), lr=0.0001)
g_optim = torch.optim.Adam(gen.parameters(), lr=0.0001)

loss_fn = torch.nn.BCELoss()

test_input = torch.randn(16, 100, device=device)  # 16个长度为100的正态随机数

# print(test_input)

D_loss = []
G_loss = []

EPOCH = 30
# 训练循环
for epoch in range(EPOCH):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)  # len(dataloader)返回批次数
    # print(count)
    # len(dataset)返回样本数
    for idx, (imgs, _) in enumerate(dataloader):
        real_input = imgs.to(device)
        size = real_input.size(0)
        random_noise = torch.randn(size, 100, device=device)

        # 1 训练判别器
        # 1.1 真实数据real_input输入dis,得到real_output
        d_optim.zero_grad()
        real_output = dis(real_input)  # 对判别器输入真实图片 real_output对真实图片的预测结果
        # 得到判别器在真实图像上面的损失
        d_real_loss = loss_fn(real_output, torch.ones_like(real_output))
        d_real_loss.backward()

        # 1.2 生成数据的输出gen_img输入dis,得到fake_output
        gen_img = gen(random_noise)
        # detach()截断生成器梯度，更新判别器梯度
        fake_output = dis(gen_img.detach())  # 判别器输入生成图片。fake_output对生成图片的预测
        # 得到判别器在生成图像上面的损失
        d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))
        d_fake_loss.backward()

        # 1.3 计算损失值
        d_loss = d_fake_loss + d_real_loss
        d_optim.step()

        # 2 训练生成器
        # 2.1 G的输出gen_img输入D，得到fake_output
        g_optim.zero_grad()
        fake_output = dis(gen_img)
        # 2.2 计算损失值，得到生成器的损失
        g_loss = loss_fn(fake_output, torch.ones_like(fake_output))
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss

        # if idx % 100 == 0 or idx == count:
        #     print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f}'.format(epoch, idx,
        #                                                                                           d_epoch_loss,
        #                                                                                           g_epoch_loss))

    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print("Epoch {}, discriminator_loss {:.3f} generator_loss {:.3f}".format(epoch, d_epoch_loss, g_epoch_loss))

    if (epoch + 1) % 10 == 0:
        torch.save(gen, os.path.join("model", 'Generator_epoch_{}.pth'.format(epoch)))
        print('Model saved.')

    if epoch == EPOCH - 1:
        gen_img_plot(gen, test_input)
