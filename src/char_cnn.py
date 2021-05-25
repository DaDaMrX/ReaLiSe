import os

import numpy as np
import torch
from torch import nn
from PIL import ImageFont


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class CharResNet(torch.nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        # input_image: bxcx32x32, output_image: bx768x1x1
        self.res_block1 = BasicBlock(in_channels, 64, stride=2)     # channels: 64, size: 16x16
        self.res_block2 = BasicBlock(64, 128, stride=2)   # channels: 128, size: 8x8
        self.res_block3 = BasicBlock(128, 256, stride=2)  # channels: 256, size: 4x4
        self.res_block4 = BasicBlock(256, 512, stride=2)  # channels: 512, size: 2x2
        self.res_block5 = BasicBlock(512, 768, stride=2)  # channels: 768, size: 1x1
        
    def forward(self, x):
        # input_shape: bxcx32x32, output_image: bx768
        # x = x.unsqueeze(1)
        h = self.res_block1(x)
        h = self.res_block2(h)
        h = self.res_block3(h)
        h = self.res_block4(h)
        h = self.res_block5(h)
        h = h.squeeze(-1).squeeze(-1)
        return h

class CharResNet1(torch.nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.res_block1 = BasicBlock(in_channels, 64, stride=2)     # channels: 64, size: 16x16
        self.res_block2 = BasicBlock(64, 128, stride=2)   # channels:  128, size: 8x8
        self.res_block3 = BasicBlock(128, 192, stride=2)  # channels: 256, size: 4x4
        self.res_block4 = BasicBlock(192, 192, stride=2)

    def forward(self, x):
        # input_shape: bxcx32x32, output_shape: bx128x8x8
        h = x
        h = self.res_block1(h)
        h = self.res_block2(h)
        h = self.res_block3(h)
        h = self.res_block4(h)
        h = h.view(h.shape[0], -1)
        return h


if __name__ == '__main__':
    # model = CharMaskResnet()

    # x = torch.rand(5, 1, 32, 32)
    # h = model(x)
    # print(x.shape)
    # print(h.shape)

    model = CharResNet(in_channels=4)

    x = torch.rand(5, 4, 32, 32)
    h = model(x)
    print(x.shape)
    print(h.shape)
