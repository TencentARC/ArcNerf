# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNRelu(nn.Module):
    """A block for conv2d-bn-relu"""

    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, groups=1, bn=True, relu=True
    ):
        super(ConvBNRelu, self).__init__()

        self.do_bn = bn
        self.do_relu = relu

        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            groups=groups,
            bias=not self.do_bn,
        )

        if self.do_bn:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        if self.do_relu:
            self.relu = nn.ReLU(inplace=True)

        self.init_weight()

    def forward(self, x):
        x = self.conv2d(x)
        if self.do_bn:
            x = self.batch_norm(x)
        if self.do_relu:
            x = self.relu(x)
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SEBlock(nn.Module):
    """ https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html """

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(
            in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True
        )
        self.up = nn.Conv2d(
            in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True
        )
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)

        return inputs * x
