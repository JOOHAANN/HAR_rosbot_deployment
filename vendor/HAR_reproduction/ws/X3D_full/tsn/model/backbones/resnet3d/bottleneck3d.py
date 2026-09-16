# -*- coding: utf-8 -*-

"""
@date: 2020/9/28 4:35 PM
@file: bottleneck3d.py
@author: zj
@description: 
"""

import torch.nn as nn
from tsn.model.layers.conv_helper import convTx1x1


class Bottleneck3d(nn.Module):
    """
    Bottleneck 3d block for ResNet3D.
    """

    def __init__(self,
                 # input channels
                 inplanes,
                 # output channels
                 planes,
                 # spatial stride
                 spatial_stride=1,
                 # whether to inflate
                 inflate=True,
                 # inflation type
                 inflate_style='3x1x1',
                 # expansion factor
                 expansion=4,
                 # conv layer type
                 conv_layer=None,
                 # norm layer type
                 norm_layer=None,
                 # activation layer type
                 act_layer=None):
        super().__init__()
        assert inflate_style in ['3x1x1', '3x3x3']
        if conv_layer is None:
            conv_layer = nn.Conv3d
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if act_layer is None:
            act_layer = nn.ReLU

        # input channels
        self.inplanes = inplanes
        # output channels
        self.planes = planes
        # spatial stride
        self.spatial_stride = spatial_stride
        # whether to inflate
        self.inflate = inflate
        # inflation type
        self.inflate_style = inflate_style
        # expansion factor
        self.expansion = expansion
        # conv layer type
        self.conv_layer = conv_layer
        # norm layer type
        self.norm_layer = norm_layer
        # activation layer type
        self.act_layer = act_layer

        if self.inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, 1, 1)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)

        # Tx1x1
        self.conv1 = convTx1x1(inplanes,
                               planes,
                               kernel_size=conv1_kernel_size,
                               padding=conv1_padding,
                               bias=False)
        self.bn1 = norm_layer(planes)

        # Tx3x3
        self.conv2 = conv_layer(planes,
                                planes,
                                kernel_size=conv2_kernel_size,
                                # whether to perform spatial downsampling
                                stride=(1, self.spatial_stride, self.spatial_stride),
                                padding=conv2_padding,
                                bias=False)
        self.bn2 = norm_layer(planes)

        # Tx1x1
        out_planes = int(planes * self.expansion)
        self.conv3 = convTx1x1(planes,
                               out_planes,
                               kernel_size=(1, 1, 1),
                               padding=(0, 0, 0),
                               bias=False)
        self.bn3 = norm_layer(out_planes)

        self.act = self.act_layer(inplace=True)
        downsample = None
        if self.spatial_stride != 1 or self.inplanes != out_planes:
            # downsampling
            # in spatial or channel dimension
            downsample = nn.Sequential(
                conv_layer(inplanes,
                           out_planes,
                           kernel_size=(1, 1, 1),
                           stride=(1, self.spatial_stride, self.spatial_stride),
                           padding=(0, 0, 0),
                           bias=False),
                norm_layer(out_planes),
            )
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out
