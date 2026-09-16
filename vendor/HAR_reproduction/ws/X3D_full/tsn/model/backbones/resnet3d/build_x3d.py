# -*- coding: utf-8 -*-

"""
@date: 2020/11/3 2:03 PM
@file: build_x3d.py
@author: zj
@description: 
"""

from tsn.model.backbones.resnet3d.resnet3d import ResNet3d
from tsn.model.backbones.resnet3d.bottleneck3d import Bottleneck3d
from tsn.model import registry

from tsn.model.layers.conv_helper import get_conv
from tsn.model.layers.pool_helper import get_pool
from tsn.model.layers.norm_helper import get_norm
from tsn.model.layers.act_helper import get_act

__all__ = ['ResNet3d', ]


def _resnet(cfg, block_layer):
    conv_layer = get_conv(cfg.MODEL.CONV_LAYER)
    pool_layer = get_pool(cfg.MODEL.POOL_LAYER)
    norm_layer = get_norm(cfg.MODEL.NORM_LAYER)
    act_layer = get_act(cfg.MODEL.ACT_LAYER)

    model = ResNet3d(
        # number of input channels
        in_channels=cfg.MODEL.BACKBONE.IN_CHANNELS,
        # number of stem channels
        base_channel=cfg.MODEL.BACKBONE.BASE_CHANNEL,
        # kernel_size of the first conv layer
        conv1_kernel=cfg.MODEL.BACKBONE.CONV1_KERNEL,
        # stride of the first conv layer
        conv1_stride=cfg.MODEL.BACKBONE.CONV1_STRIDE,
        # zero padding of the first conv layer
        conv1_padding=cfg.MODEL.BACKBONE.CONV1_PADDING,
        # whether to use the first pooling layer
        with_pool1=cfg.MODEL.BACKBONE.WITH_POOL1,
        # whether to use the second pooling layer
        with_pool2=cfg.MODEL.BACKBONE.WITH_POOL2,
        # number of blocks per stage, e.g. R50
        stage_blocks=cfg.MODEL.BACKBONE.STAGE_BLOCKS,
        # output channels of the first conv layer in each stage
        res_planes=cfg.MODEL.BACKBONE.RES_PLANES,
        # expansion factor, e.g. Bottleneck
        expansion=cfg.MODEL.BACKBONE.EXPANSION,
        # spatial stride
        spatial_strides=cfg.MODEL.BACKBONE.SPATIAL_STRIDES,
        # whether to inflate
        inflates=cfg.MODEL.BACKBONE.INFLATES,
        # inflation type
        inflate_style=cfg.MODEL.BACKBONE.INFLATE_STYLE,
        # conv layer type
        conv_layer=conv_layer,
        # pooling layer type
        pool_layer=pool_layer,
        # norm layer type
        norm_layer=norm_layer,
        # activation layer type
        act_layer=act_layer,
        # block type
        block_layer=block_layer,
        # whether to zero-init the residual branch
        zero_init_residual=cfg.MODEL.BACKBONE.ZERO_INIT_RESIDUAL,
        # whether to apply partial BN
        partial_bn=cfg.MODEL.BACKBONE.PARTIAL_BN,
    )
    return model


@registry.BACKBONE.register('X3D')
def x3d(cfg):
    return _resnet(cfg, Bottleneck3d)
