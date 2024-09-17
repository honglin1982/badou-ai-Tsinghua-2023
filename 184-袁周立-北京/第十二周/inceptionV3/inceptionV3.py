import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models


class Conv2dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding='same'):
        super(Conv2dBn, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchNorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchNorm(x)
        x = F.relu(x)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_out_channels):
        # 四个尺度：
        # 1x1
        # 1x1 -> 5x5
        # 1x1 -> 3x3 -> 3x3
        # pool -> 1x1
        super(InceptionA, self).__init__()

        self.conv2d_1x1 = Conv2dBn(in_channels, 64, (1, 1))
        self.conv2d_5x5 = nn.Sequential(
            Conv2dBn(in_channels, 48, (1, 1)),
            Conv2dBn(48, 64, (5, 5))
        )
        self.conv2d_3x3 = nn.Sequential(
            Conv2dBn(in_channels, 64, (1, 1)),
            Conv2dBn(64, 96, (3, 3)),
            Conv2dBn(96, 96, (3, 3))
        )
        self.pool_1x1 = nn.Sequential(
            nn.AvgPool2d((3, 3), stride=(1, 1), padding=1),
            Conv2dBn(in_channels, pool_out_channels, (1, 1))
        )

    def forward(self, x):   # torch支持的是chw，所以在dim=1维度进行拼接
        branch1x1 = self.conv2d_1x1(x)
        branch5x5 = self.conv2d_5x5(x)
        branch3x3 = self.conv2d_3x3(x)
        branch_pool = self.pool_1x1(x)
        return torch.cat([branch1x1, branch5x5, branch3x3, branch_pool], dim=1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        # 三个尺度：
        # 3x3
        # 1x1 -> 3x3 -> 3x3
        # pool
        super(InceptionB, self).__init__()

        self.conv2d_3x3 = Conv2dBn(in_channels, 384, (3, 3), stride=(2, 2), padding='valid')
        self.conv2d_1x1_3x3 = nn.Sequential(
            Conv2dBn(in_channels, 64, (1, 1)),
            Conv2dBn(64, 96, (3, 3)),
            Conv2dBn(96, 96, (3, 3), stride=(2, 2), padding='valid')
        )
        self.pool = nn.MaxPool2d((3, 3), stride=(2, 2))

    def forward(self, x):
        branch3x3 = self.conv2d_3x3(x)
        branch1x1_3x3 = self.conv2d_1x1_3x3(x)
        branch_pool = self.pool(x)
        return torch.cat([branch3x3, branch1x1_3x3, branch_pool], dim=1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, mid_channels):
        # 四个尺度：
        # 1x1
        # 1x1 -> 1x7 -> 7x1
        # 1x1 -> 7x1 -> 1x7 -> 7x1 -> 1x7
        # pool -> 1x1
        super(InceptionC, self).__init__()

        self.conv2d_1x1 = Conv2dBn(in_channels, 192, (1, 1))
        self.conv2d_1x7_7x1 = nn.Sequential(
            Conv2dBn(in_channels, mid_channels, (1, 1)),
            Conv2dBn(mid_channels, mid_channels, (1, 7)),
            Conv2dBn(mid_channels, 192, (7, 1))
        )
        self.conv2d_7x1_1x7_7x1_1x7 = nn.Sequential(
            Conv2dBn(in_channels, mid_channels, (1, 1)),
            Conv2dBn(mid_channels, mid_channels, (7, 1)),
            Conv2dBn(mid_channels, mid_channels, (1, 7)),
            Conv2dBn(mid_channels, mid_channels, (7, 1)),
            Conv2dBn(mid_channels, 192, (1, 7)),
        )
        self.pool_1x1 = nn.Sequential(
            nn.AvgPool2d((3, 3), stride=(1, 1), padding=1),
            Conv2dBn(in_channels, 192, (1, 1))
        )

    def forward(self, x):
        branch1x1 = self.conv2d_1x1(x)
        branch1x7_7x1 = self.conv2d_1x7_7x1(x)
        branch7x1_1x7_7x1_1x7 = self.conv2d_7x1_1x7_7x1_1x7(x)
        branch_pool = self.pool_1x1(x)
        return torch.cat([branch1x1, branch1x7_7x1, branch7x1_1x7_7x1_1x7, branch_pool], dim=1)


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        # 三个尺度：
        # 1x1 -> 3x3
        # 1x1 -> 1x7 -> 7x1 -> 3x3
        # pool
        super(InceptionD, self).__init__()

        self.conv2d_1x1_3x3 = nn.Sequential(
            Conv2dBn(in_channels, 192, (1, 1)),
            Conv2dBn(192, 320, (3, 3), stride=(2, 2), padding='valid'),
        )
        self.conv2d_1x7_7x1 = nn.Sequential(
            Conv2dBn(in_channels, 192, (1, 1)),
            Conv2dBn(192, 192, (1, 7)),
            Conv2dBn(192, 192, (7, 1)),
            Conv2dBn(192, 192, (3, 3), stride=(2, 2), padding='valid'),
        )
        self.pool = nn.MaxPool2d((3, 3), stride=(2, 2))

    def forward(self, x):
        branch_1x1_3x3 = self.conv2d_1x1_3x3(x)
        branch_1x7_7x1 = self.conv2d_1x7_7x1(x)
        branch_pool = self.pool(x)
        return torch.cat([branch_1x1_3x3, branch_1x7_7x1, branch_pool], dim=1)


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        # 四个尺度：
        # 1x1
        # 1x1 -> 1x3 3x1
        # 1x1 -> 3x3 -> 1x3 3x1
        # pool -> 1x1
        super(InceptionE, self).__init__()

        self.conv2d_1x1 = Conv2dBn(in_channels, 320, (1, 1))

        self.conv2d_1x1_1331_1 = Conv2dBn(in_channels, 384, (1, 1))
        self.conv2d_1x1_1331_21 = Conv2dBn(384, 384, (1, 3))
        self.conv2d_1x1_1331_22 = Conv2dBn(384, 384, (3, 1))

        self.conv2d_1x1_3x3_1331_1 = Conv2dBn(in_channels, 448, (1, 1))
        self.conv2d_1x1_3x3_1331_2 = Conv2dBn(448, 384, (3, 3))
        self.conv2d_1x1_3x3_1331_31 = Conv2dBn(384, 384, (1, 3))
        self.conv2d_1x1_3x3_1331_32 = Conv2dBn(384, 384, (3, 1))

        self.pool = nn.Sequential(
            nn.AvgPool2d((3, 3), stride=(1, 1), padding=1),
            Conv2dBn(in_channels, 192, (1, 1))
        )

    def forward(self, x):
        branch_1x1 = self.conv2d_1x1(x)

        branch_1x1_1331_1 = self.conv2d_1x1_1331_1(x)
        branch_1x1_1331_21 = self.conv2d_1x1_1331_21(branch_1x1_1331_1)
        branch_1x1_1331_22 = self.conv2d_1x1_1331_22(branch_1x1_1331_1)
        branch_1x1_1331 = torch.cat([branch_1x1_1331_21, branch_1x1_1331_22], dim=1)

        branch_1x1_3x3_1331_1 = self.conv2d_1x1_3x3_1331_1(x)
        branch_1x1_3x3_1331_2 = self.conv2d_1x1_3x3_1331_2(branch_1x1_3x3_1331_1)
        branch_1x1_3x3_1331_31 = self.conv2d_1x1_3x3_1331_31(branch_1x1_3x3_1331_2)
        branch_1x1_3x3_1331_32 = self.conv2d_1x1_3x3_1331_32(branch_1x1_3x3_1331_2)
        branch_1x1_3x3_1331 = torch.cat([branch_1x1_3x3_1331_31, branch_1x1_3x3_1331_32], dim=1)

        branch_pool = self.pool(x)
        return torch.cat([branch_1x1, branch_1x1_1331, branch_1x1_3x3_1331, branch_pool], dim=1)


class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.conv2d_bns = nn.Sequential(
            Conv2dBn(3, 32, (3, 3), stride=(2, 2), padding='valid'),    # (3, 299, 299) => (32, 149, 149)
            Conv2dBn(32, 32, (3, 3), padding='valid'),                  # => (32, 147, 147)
            Conv2dBn(32, 64, (3, 3)),                                   # => (64, 147, 147)
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),            # => (64, 73, 73)
            Conv2dBn(64, 80, (1, 1), padding='valid'),                  # => (80, 73, 73)
            Conv2dBn(80, 192, (3, 3), padding='valid'),                 # => (192, 71, 71)
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),            # => (192, 35, 35)
        )
        self.inceptionA1 = InceptionA(192, 32)                  # => (256, 35, 35)
        self.inceptionA2 = InceptionA(256, 64)                  # => (288, 35, 35)
        self.inceptionA3 = InceptionA(288, 64)                  # => (288, 35, 35)

        self.inceptionB = InceptionB(288)                # => (768, 17, 17)

        self.inceptionC1 = InceptionC(768, 128)          # => (768, 17, 17)
        self.inceptionC2 = InceptionC(768, 160)          # => (768, 17, 17)
        self.inceptionC3 = InceptionC(768, 160)          # => (768, 17, 17)
        self.inceptionC4 = InceptionC(768, 192)          # => (768, 17, 17)

        self.inceptionD = InceptionD(768)         # => (1280, 8, 8)

        self.inceptionE1 = InceptionE(1280)        # => (2048, 8, 8)
        self.inceptionE2 = InceptionE(2048)        # => (2048, 8, 8)

        self.linear = nn.Linear(2048, 1000)

    def forward(self, x):   # input_shape: (3, 299, 299)
        x = self.conv2d_bns(x)

        x = self.inceptionA1(x)
        x = self.inceptionA2(x)
        x = self.inceptionA3(x)

        x = self.inceptionB(x)

        x = self.inceptionC1(x)
        x = self.inceptionC2(x)
        x = self.inceptionC3(x)
        x = self.inceptionC4(x)

        x = self.inceptionD(x)

        x = self.inceptionE1(x)
        x = self.inceptionE2(x)

        # 平均池化后全连接
        x = F.avg_pool2d(x, kernel_size=8)         # => (2048, 1, 1)
        x = x.squeeze()
        x = self.linear(x)
        x = F.softmax(x, dim=-1)

        return x


if __name__ == '__main__':
    inceptionV3_model = InceptionV3()

    # ！！！ 模型参数加载不上去，因为state_dict的key对应不上，似乎层的命名是强要求的，虽然可以通过strict=False让代码跑起来，但实际上参数没加载正确
    # ！！！ 模型参数加载不上去，因为state_dict的key对应不上，似乎层的命名是强要求的，虽然可以通过strict=False让代码跑起来，但实际上参数没加载正确
    # ！！！ 所以可能模型结构与论文原版有出入，但通过这个pth文件不好验证，因为层的变量命名以及Sequential什么的，肯定没有和torch内置的inceptionV3一致，导致参数加载不正确，
    # ！！！ 但基本结构和论文原版应该没啥问题，达到对inceptionV3结构的理解加深的目的也够了
    inceptionV3_model.load_state_dict(torch.load('./inception_v3_google-0cc3c7bd.pth'), strict=False)

    img_path = 'elephant.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))
    x_input = img / 255.0

    x_input = np.transpose(x_input, (2, 0, 1))
    x_input = np.expand_dims(x_input, axis=0)
    x_input = torch.tensor(x_input, dtype=torch.float32)

    predict = inceptionV3_model(x_input)

    result = torch.argmax(predict)

    print(predict)
    print(result)
    #
    # print(inceptionV3_model.state_dict().keys())
    #
    # model = torchvision.models.inception_v3()
    # print(model.state_dict().keys())
    #
    # keys1 = inceptionV3_model.state_dict().keys()
    # keys2 = model.state_dict().keys()
    # print(len(keys1), len(keys2))