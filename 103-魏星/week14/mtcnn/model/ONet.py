# _*_ coding : utf-8 _*_
# @Time : 2023/8/17 17:30
# @Author : weixing
# @FileName : ONet
# @Project : cv

import torch.nn as nn


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.fpn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),  # (48,48,3)->(46,46,32)
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True),  # (46,46,32) -> (23,23,28)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),  # (23,23,32)->(21,21,64)
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True),  # (21,21,64)->(10,10,64)

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),  # (10,10,64)->(8,8,64)
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),  # (8,8,64)->(4,4,64)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),  # (4,4,64)->(3,3,128)
            nn.PReLU()
        )

        self.flatten = nn.Flatten()  # (3,3,128) -> (1152)
        self.fc = nn.Linear(in_features=3 * 3 * 128, out_features=256)  # (1152)->(256)

        self.face_class_fc = nn.Linear(256, 2)  # (256)->(2)
        self.bbox_reg_fc = nn.Linear(256, 4)  # (256)->(4)
        self.facial_landmark_fc = nn.Linear(256, 10)  # (256)->(10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.fpn(x)
        x = self.flatten(x)
        x = self.fc(x)

        class_out = self.face_class_fc(x)

        bbox_out = self.bbox_reg_fc(x)

        landmark_out = self.facial_landmark_fc(x)

        return class_out, bbox_out, landmark_out


model = ONet()
print(model)