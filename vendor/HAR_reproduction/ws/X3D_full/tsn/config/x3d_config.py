# -*- coding: utf-8 -*-

"""
@date: 2020/11/3 2:42 PM
@file: x3d_config.py
@author: zj
@description: 
"""


def add_custom_config(_C):
    # Inflated 3D ConvNet (I3D).

    # number of channels of the 5th conv layer
    _C.MODEL.HEAD.CONV5_CHANNELS = 192
    # kernel_size of the 5th pooling layer
    _C.MODEL.HEAD.POOL5_KERNEL = (1, 4, 4)