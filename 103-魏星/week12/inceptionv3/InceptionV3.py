# _*_ coding : utf-8 _*_
# @Time : 2023/8/1 16:26
# @Author : weixing
# @FileName : InceptionV3
# @Project : cv

import torch
import torch.nn as nn

'''
InceptionV3层数：
骨干网络：
5层卷积+
3(block1_module1)+3(block1_module2)+3(block1_module3)+
3(block2_module1)+5(block2_module2)+5(block2_module3)+5(block2_module4)+5(block2_module5)+
4(block3_module1)+3(block3_module2)+3(block3_module3)+
maxPool
=47层

分类：
+dropout+(conv+bn+relu)
'''


def ConvBnRelu(input_channel, output_channel, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True)
    )


def ConvBnReluTuple(input_channel, output_channel, kernel_sizes=(3, 3), strides=(1, 1), paddings=(0, 0)):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size=kernel_sizes, stride=strides, padding=paddings),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True)
    )


'''
block1_module1  
共4个分支 input[N,35,35,192] out [N,35,35,256] 
branch1: conv(1x1,64,stride=1)
branch2: conv(1x1,48,stride=1)  conv(5x5,64,stride=1,padding=2)
branch3: conv(1x1,64,stride=1)  conv(3x3,96,stride=1,padding=1)   conv(3x3,96,stride=1,padding=2)
branch4: avg_pool(3x3,stride=1,padding=1)  conv(1x1,32,stride=1)

block1_module2  
共4个分支 input[N,35,35,256] out [N,35,35,288] 
branch1: conv(1x1,64,stride=1)
branch2: conv(1x1,48,stride=1)  conv(5x5,64,stride=1,padding=2)
branch3: conv(1x1,64,stride=1)  conv(3x3,96,stride=1,padding=1)   conv(3x3,96,stride=1,padding=2)
branch4: avg_pool(3x3,stride=1,padding=1)  conv(1x1,64,stride=1)

block1_module3  
共4个分支 input[N,35,35,288] out [N,35,35,288] 
branch1: conv(1x1,64,stride=1)
branch2: conv(1x1,48,stride=1)  conv(5x5,64,stride=1,padding=2)
branch3: conv(1x1,64,stride=1)  conv(3x3,96,stride=1,padding=1)   conv(3x3,96,stride=1,padding=2)
branch4: avg_pool(3x3,stride=1,padding=1)  conv(1x1,64,stride=1)
'''


class Block1_Module(nn.Module):
    def __init__(self, input_channel, pool_channel):
        super(Block1_Module, self).__init__()

        self.branch1_1x1 = ConvBnRelu(input_channel, 64,
                                      kernel_size=1)  # input [N,35,35,input_channel] out [N,35,35,64]

        self.branch2_5x5 = nn.Sequential(
            ConvBnRelu(input_channel, 48, kernel_size=1),  # input [N,35,35,input_channel] out [N,35,35,48]
            ConvBnRelu(48, 64, kernel_size=5, stride=1, padding=2)  # input [N,35,35,48] out [N,35,35,64]
        )

        self.branch3_3x3x2 = nn.Sequential(
            ConvBnRelu(input_channel, 64, kernel_size=1, stride=1, padding=0),
            # input [N,35,35,input_channel] out [N,35,35,64]
            ConvBnRelu(64, 96, kernel_size=3, stride=1, padding=1),  # input [N,35,35,64] out [N,35,35,96]
            ConvBnRelu(96, 96, kernel_size=3, stride=1, padding=1)  # input [N,35,35,96] out [N,35,35,96]
        )

        self.branch4_avgPool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            # input [N,35,35,input_channel] out [N,35,35,input_channel]
            ConvBnRelu(input_channel, pool_channel, kernel_size=1)
            # input [N,35,35,input_channel]  out [N,35,35,pool_channel]
        )

    def forward(self, x):
        branch1 = self.branch1_1x1(x)
        branch2 = self.branch2_5x5(x)
        branch3 = self.branch3_3x3x2(x)
        branch4 = self.branch4_avgPool(x)

        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs


'''
block2_module1  
共3个分支 input[N,35,35,288] out [N,17,17,768] 
branch1: conv(3x3,384,stride=2)
branch2: conv(1x1,64,stride=1)  conv(3x3,96,stride=1,padding=1)   conv(3x3,96,stride=2)
branch3: max_pool(3x3,stride=2) 
'''


class Block2_ModuleA(nn.Module):
    def __init__(self, input_channel):
        super(Block2_ModuleA, self).__init__()

        self.branch1_3x3 = ConvBnRelu(input_channel, 384, kernel_size=3, stride=2)
        # input [N,35,35,input_channel] out [N,17,17,384]

        self.branch2_3x3x2 = nn.Sequential(
            ConvBnRelu(input_channel, 64, kernel_size=1),  # input [N,35,35,input_channel] out [N,35,35,64]
            ConvBnRelu(64, 96, kernel_size=3, padding=1),  # input [N,35,35,64] out [N,35,35,96]
            ConvBnRelu(96, 96, kernel_size=3, stride=2)  # input [N,35,35,96] out [N,17,17,96]
        )

        self.branch3_maxPool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2)  # input [N,35,35,input_channel] out [N,17,17,input_channel]
        )

    def forward(self, x):
        branch1 = self.branch1_3x3(x)
        branch2 = self.branch2_3x3x2(x)
        branch3 = self.branch3_maxPool(x)

        outputs = torch.cat([branch1, branch2, branch3], dim=1)
        return outputs


'''
block2_module2  
共4个分支 input[N,17,17,768] out [N,17,17,768] 
branch1: conv(1x1,192,stride=1)
branch2: conv(1x1,128,stride=1)  conv(1x7,128,stride=1) conv(7x1,192,stride=1)
branch3: conv(1x1,128,stride=1)  conv(7x1,128,stride=1) conv(1x7,128,stride=1) conv(7x1,128,stride=1) conv(1x7,192,stride=1)
branch4: avg_pool(3x3,stride=1,padding=1)  conv(1x1,192,stride=1)


block2_module3  block2_module4
共4个分支 input[N,17,17,768] out [N,17,17,768] 
branch1: conv(1x1,192,stride=1)
branch2: conv(1x1,160,stride=1)  conv(1x7,160,stride=1) conv(7x1,192,stride=1)
branch3: conv(1x1,160,stride=1)  conv(7x1,160,stride=1) conv(1x7,160,stride=1) conv(7x1,160,stride=1) conv(1x7,192,stride=1)
branch4: avg_pool(3x3,stride=1,padding=1)  conv(1x1,192,stride=1)


block2_module5
共4个分支 input[N,17,17,768] out [N,17,17,768] 
branch1: conv(1x1,192,stride=1)
branch2: conv(1x1,192,stride=1)  conv(1x7,192,stride=1) conv(7x1,192,stride=1)
branch3: conv(1x1,192,stride=1)  conv(7x1,192,stride=1) conv(1x7,192,stride=1) conv(7x1,192,stride=1) conv(1x7,192,stride=1)
branch4: avg_pool(3x3,stride=1,padding=1)  conv(1x1,192,stride=1)
'''


class Block2_ModuleB(nn.Module):
    def __init__(self, input_channel, middle_channel):
        super(Block2_ModuleB, self).__init__()

        self.branch1_1x1 = ConvBnRelu(input_channel, 192, kernel_size=1)  # input[N,17,17,768] out[N,17,17,192]

        self.branch2_7x7_a = nn.Sequential(
            ConvBnRelu(input_channel, middle_channel, kernel_size=1),  # input[N,17,17,768] out[N,17,17,middle_channel]
            ConvBnReluTuple(middle_channel, middle_channel, kernel_sizes=(1, 7), paddings=(0, 3)),
            # input[N,17,17,middle_channel] out[N,17,17,middle_channel]
            ConvBnReluTuple(middle_channel, 192, kernel_sizes=(7, 1), paddings=(3, 0))
            # input[N,17,17,middle_channel] out[N,17,17,192]
        )

        self.branch3_7x7_b = nn.Sequential(
            ConvBnRelu(input_channel, middle_channel, kernel_size=1),  # input[N,17,17,768] out[N,17,17,middle_channel]
            ConvBnReluTuple(middle_channel, middle_channel, kernel_sizes=(7, 1), paddings=(3, 0)),
            # input[N,17,17,middle_channel] out[N,17,17,middle_channel]
            ConvBnReluTuple(middle_channel, middle_channel, kernel_sizes=(1, 7), paddings=(0, 3)),
            # input[N,17,17,middle_channel] out[N,17,17,middle_channel]
            ConvBnReluTuple(middle_channel, middle_channel, kernel_sizes=(7, 1), paddings=(3, 0)),
            # input[N,17,17,middle_channel] out[N,17,17,middle_channel]
            ConvBnReluTuple(middle_channel, 192, kernel_sizes=(1, 7), paddings=(0, 3))
            # input[N,17,17,middle_channel] out[N,17,17,192]
        )

        self.branch4_avgPool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),  # input[N,17,17,768] out[N,17,17,768]
            ConvBnRelu(768, 192, kernel_size=1)  # input[N,17,17,192] out[N,17,17,192]
        )

    def forward(self, x):
        branch1 = self.branch1_1x1(x)
        branch2 = self.branch2_7x7_a(x)
        branch3 = self.branch3_7x7_b(x)
        branch4 = self.branch4_avgPool(x)

        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs


'''
block3_module1  
共3个分支 input[N,17,17,768] out [N,8,8,1280] 
branch1: conv(1x1,192,stride=1)  conv(3x3,320,stride=2)
branch2: conv(1x1,192,stride=1)  conv(1x7,192,stride=1)   conv(7x1,192,stride=1) conv(3x3,192,stride=2)
branch3: max_pool(3x3,stride=2) 
'''


class Block3_ModuleA(nn.Module):
    def __init__(self, input_channel):
        super(Block3_ModuleA, self).__init__()

        self.branch1_3x3 = nn.Sequential(
            ConvBnRelu(input_channel, 192, kernel_size=1),  # input[N,17,17,768]  out[N,17,17,192]
            ConvBnRelu(192, 320, kernel_size=3, stride=2)  # input [N,17,17,192] out [N,8,8,320]
        )

        self.branch2_7x7_a = nn.Sequential(
            ConvBnRelu(input_channel, 192, kernel_size=1),  # input[N,17,17,768] out[N,17,17,192]
            ConvBnReluTuple(192, 192, kernel_sizes=(1, 7), paddings=(0, 3)),
            # input[N,17,17,192] out[N,17,17,192]
            ConvBnReluTuple(192, 192, kernel_sizes=(7, 1), paddings=(3, 0)),
            # input[N,17,17,192] out[N,17,17,192]
            ConvBnRelu(192, 192, kernel_size=3, stride=2)  # input[N,17,17,192] out[N,17,17,192]
        )

        self.branch3_maxPool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2)  # input [N,17,17,input_channel] out [N,17,17,input_channel]
        )

    def forward(self, x):
        branch1 = self.branch1_3x3(x)
        branch2 = self.branch2_7x7_a(x)
        branch3 = self.branch3_maxPool(x)

        outputs = torch.cat([branch1, branch2, branch3], dim=1)
        return outputs


'''
block3_module2 
共4个分支 input[N,8,8,1280]  out [N,8,8,2048] 
branch1: conv(1x1,320,stride=1)
branch2: conv(1x1,384,stride=1)  conv(1x3,384,stride=1)+conv(3x1,384,stride=1)
branch3: conv(1x1,448,stride=1)  conv(3x3,384,stride=1)  conv(1x3,384,stride=1)+conv(3x1,384,stride=1)
branch4: avg_pool(3x3,stride=1)  conv(1x1,192,stride=1)

block3_module3
共4个分支 input[N,8,8,2048]  out [N,8,8,2048] 
branch1: conv(1x1,320,stride=1)
branch2: conv(1x1,384,stride=1)  conv(1x3,384,stride=1)+conv(3x1,384,stride=1)
branch3: conv(1x1,448,stride=1)  conv(3x3,384,stride=1)  conv(1x3,384,stride=1)+conv(3x1,384,stride=1)
branch4: avg_pool(3x3,stride=1)  conv(1x1,192,stride=1)
'''


class Block3_ModuleB(nn.Module):
    def __init__(self, input_channel):
        super(Block3_ModuleB, self).__init__()

        self.branch1_1x1 = ConvBnRelu(input_channel, 320, kernel_size=1)  # input[N,8,8,input_channel]  out [N,8,8,320]

        self.branch2_a_1x3 = nn.Sequential(
            ConvBnRelu(input_channel, 384, kernel_size=1),  # input[N,8,8,input_channel]  out [N,8,8,384]
            ConvBnReluTuple(384, 384, kernel_sizes=(1, 3), paddings=(0, 1))  # input[N,8,8,384]  out [N,8,8,384]
        )

        self.branch2_b_3x1 = nn.Sequential(
            ConvBnRelu(input_channel, 384, kernel_size=1),  # input[N,8,8,input_channel]  out [N,8,8,384]
            ConvBnReluTuple(384, 384, kernel_sizes=(3, 1), paddings=(1, 0))  # input[N,8,8,384]  out [N,8,8,384]
        )

        self.branch3_a_1x3 = nn.Sequential(
            ConvBnRelu(input_channel, 448, kernel_size=1),  # input[N,8,8,input_channel]  out [N,8,8,448]
            ConvBnRelu(448, 384, kernel_size=3, padding=1),  # input[N,8,8,448]  out [N,8,8,384]
            ConvBnReluTuple(384, 384, kernel_sizes=(1, 3), paddings=(0, 1))  # input[N,8,8,384]  out [N,8,8,384]
        )

        self.branch3_b_3x1 = nn.Sequential(
            ConvBnRelu(input_channel, 448, kernel_size=1),  # input[N,8,8,input_channel]  out [N,8,8,448]
            ConvBnRelu(448, 384, kernel_size=3, padding=1),  # input[N,8,8,448]  out [N,8,8,384]
            ConvBnReluTuple(384, 384, kernel_sizes=(3, 1), paddings=(1, 0))  # input[N,8,8,384]  out [N,8,8,384]
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),  # input[N,8,8,input_channel]  out [N,8,8,input_channel]
            ConvBnRelu(input_channel, 192, kernel_size=1)  # input[N,8,8,input_channel]  out [N,8,8,192]
        )

    def forward(self, x):
        branch1 = self.branch1_1x1(x)

        branch2_a = self.branch2_a_1x3(x)
        branch2_b = self.branch2_b_3x1(x)
        branch2 = torch.cat([branch2_a, branch2_b], dim=1)

        branch3_a = self.branch3_a_1x3(x)
        branch3_b = self.branch3_b_3x1(x)
        branch3 = torch.cat([branch3_a, branch3_b], dim=1)

        branch4 = self.branch4(x)

        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs


class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV3, self).__init__()

        self.conv_1 = ConvBnRelu(3, 32, kernel_size=3, stride=2)  # input [N,299,299,3]  output[N,149,149,32]
        self.conv_2 = ConvBnRelu(32, 32, kernel_size=3)  # input [N,149,149,32] output[N,147,147,32]
        self.conv_3 = ConvBnRelu(32, 64, kernel_size=3, padding=1)  # input [N,147,147,32] output [N,147,147,64]
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)  # input [N,147,147,64]  output [N,73,73,64]
        self.conv_4 = ConvBnRelu(64, 80, kernel_size=1, stride=1)  # input [N,73,73,64]  output [N,73,73,80]
        self.conv_5 = ConvBnRelu(80, 192, kernel_size=3, stride=1)  # input [N,73,73,80]  output [N,71,71,192]
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2)  # input [N,71,71,192]  output [N,35,35,192]

        self.block1_module1 = Block1_Module(192, 32)  # input [N,35,35,192]  out [N,35,35,256]
        self.block1_module2 = Block1_Module(256, 64)  # input [N,35,35,256]  out [N,35,35,288]
        self.block1_module3 = Block1_Module(288, 64)  # input [N,35,35,288]  out [N,35,35,288]

        self.block2_module1 = Block2_ModuleA(288)  # input [N,35,35,288]  out [N,17,17,768]
        self.block2_module2 = Block2_ModuleB(768, 128)  # input [N,17,17,768]  out [N,17,17,768]
        self.block2_module3 = Block2_ModuleB(768, 160)  # input [N,17,17,768]  out [N,17,17,768]
        self.block2_module4 = Block2_ModuleB(768, 160)  # input [N,17,17,768]  out [N,17,17,768]
        self.block2_module5 = Block2_ModuleB(768, 192)  # input [N,17,17,768]  out [N,17,17,768]

        self.block3_module1 = Block3_ModuleA(768)  # input [N,17,17,768]  out [N,8,8,1280]
        self.block3_module2 = Block3_ModuleB(1280)  # input [N,8,8,1280]  out [N,8,8,2048]
        self.block3_module3 = Block3_ModuleB(2048)  # input [N,8,8,2048]  out [N,8,8,2048]

        self.max_pool_3 = nn.MaxPool2d(kernel_size=8)  # input [N,8,8,2048]  out [N,1,1,2048]

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            ConvBnRelu(2048, num_classes, kernel_size=1)  # input [N,1,1,2048]  out [N,1,1,num_classes]
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.max_pool_1(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.max_pool_2(x)

        x = self.block1_module1(x)
        x = self.block1_module2(x)
        x = self.block1_module3(x)

        x = self.block2_module1(x)
        x = self.block2_module2(x)
        x = self.block2_module3(x)
        x = self.block2_module4(x)
        x = self.block2_module5(x)

        x = self.block3_module1(x)
        x = self.block3_module2(x)
        x = self.block3_module3(x)

        x = self.max_pool_3(x)

        x = self.classifier(x)

        output = torch.squeeze(x)  # input [N,num_classes,1,1]  out [N, num_classes]

        return output
