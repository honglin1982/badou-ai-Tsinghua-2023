import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class DepthWiseConvBlock(nn.Module):
    def __init__(self, in_channels, point_wise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
        super(DepthWiseConvBlock, self).__init__()
        self.conv_dw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * depth_multiplier, kernel_size=(3, 3),
                      stride=strides, padding=1, groups=in_channels * depth_multiplier),
            nn.BatchNorm2d(in_channels*depth_multiplier),
            nn.ReLU6(inplace=True)
        )
        self.conv_pw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * depth_multiplier, out_channels=point_wise_conv_filters,
                      kernel_size=(1, 1), padding=0, stride=(1, 1)),
            nn.BatchNorm2d(point_wise_conv_filters),
            nn.ReLU6(inplace=True)
        )

    def forward(self, inputs):
        x = self.conv_dw(inputs)
        x = self.conv_pw(x)
        return x


class MobileNet(nn.Module):
    """ chw """

    def __init__(self, in_channels=3, depth_multiplier=1, dropout=1e-3, num_classes=1000):
        super(MobileNet, self).__init__()
        # 224,224,3 -> 112,112,32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # depth wise , point wise
        self._depth_wise_conv_blocks = nn.Sequential(
            # 112,112,32 -> 112,112,64
            DepthWiseConvBlock(32, 64, depth_multiplier=depth_multiplier, strides=(1, 1), block_id=1),
            # 112,112,64 -> 56,56,128
            DepthWiseConvBlock(64, 128, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=2),
            # 56,56,128 -> 56,56,128
            DepthWiseConvBlock(128, 128, depth_multiplier=depth_multiplier, strides=(1, 1), block_id=3),
            # 56,56,128 -> 28,28,256
            DepthWiseConvBlock(128, 256, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=4),
            # 28,28,256 -> 28,28,256
            DepthWiseConvBlock(256, 256, depth_multiplier=depth_multiplier, strides=(1, 1), block_id=5),
            # 28,28,256 -> 14,14,512
            DepthWiseConvBlock(256, 512, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=6),
            # 14,14,512 -> 14,14,512
            DepthWiseConvBlock(512, 512, depth_multiplier=depth_multiplier, strides=(1, 1), block_id=7),
            DepthWiseConvBlock(512, 512, depth_multiplier=depth_multiplier, strides=(1, 1), block_id=8),
            DepthWiseConvBlock(512, 512, depth_multiplier=depth_multiplier, strides=(1, 1), block_id=9),
            DepthWiseConvBlock(512, 512, depth_multiplier=depth_multiplier, strides=(1, 1), block_id=10),
            DepthWiseConvBlock(512, 512, depth_multiplier=depth_multiplier, strides=(1, 1), block_id=11),
            # 14,14,512 -> 7,7,1024
            DepthWiseConvBlock(512, 1024, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=12),
            DepthWiseConvBlock(1024, 1024, depth_multiplier=depth_multiplier, strides=(1, 1), block_id=13),
        )
        # 7,7,1024 -> 1,1,1024
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)
        # self.init_params()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self._depth_wise_conv_blocks(x)
        x = self.pool(x)
        x = self.dropout(x)
        # print(x.shape)  # torch.Size([1, 1024, 1, 1])
        x = x.view(x.size(0), -1)
        # print(x.shape)  # torch.Size([1, 1024])
        x = self.linear(x)
        # print(x.shape)  # torch.Size([1, 1000])
        x = self.softmax(x)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    model = MobileNet()
    # print(model)
    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out)
    print(torch.argmax(out))
    print(out.shape)
    pass



