#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Inflated 3D ConvNet (I3D).

    # stem channels
    _C.MODEL.BACKBONE.BASE_CHANNEL = 64
    # kernel_size of the first conv layer
    _C.MODEL.BACKBONE.CONV1_KERNEL = (1, 7, 7)
    # stride of the first conv layer
    _C.MODEL.BACKBONE.CONV1_STRIDE = (2, 2, 2)
    # zero padding of the first conv layer
    _C.MODEL.BACKBONE.CONV1_PADDING = (0, 3, 3)
    # whether to use the first pooling layer
    _C.MODEL.BACKBONE.WITH_POOL1 = True
    # kernel_size of the first pooling layer
    _C.MODEL.BACKBONE.POOL1_KERNEL = (3, 3, 3)
    # stride of the first pooling layer
    _C.MODEL.BACKBONE.POOL1_STRIDE = (2, 2, 2)
    # whether to use the second pooling layer
    _C.MODEL.BACKBONE.WITH_POOL2 = True
    # kernel_size of the second pooling layer
    _C.MODEL.BACKBONE.POOL2_KERNEL = (3, 1, 1)
    # stride of the second pooling layer
    _C.MODEL.BACKBONE.POOL2_STRIDE = (2, 1, 1)
    # number of blocks in each stage, e.g. R50
    _C.MODEL.BACKBONE.STAGE_BLOCKS = (3, 4, 6, 3)
    # output channels of the first conv layer in each stage block
    _C.MODEL.BACKBONE.RES_PLANES = [64, 128, 256, 512]
    # expansion factor, e.g. Bottleneck
    _C.MODEL.BACKBONE.EXPANSION = 4.
    # spatial strides
    _C.MODEL.BACKBONE.SPATIAL_STRIDES = (1, 2, 2, 2)
    # whether to inflate
    _C.MODEL.BACKBONE.INFLATES = (0, 0, 0, 0)
    # inflation style
    _C.MODEL.BACKBONE.INFLATE_STYLE = '3x1x1'
