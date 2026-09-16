# -*- coding: utf-8 -*-

"""
@date: 2020/11/3 9:40 AM
@file: build_resnet3d.py
@author: zj
@description: 
"""

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torchvision.models.utils import load_state_dict_from_url

from .resnet3d import ResNet3d
from .bottleneck3d import Bottleneck3d
from tsn.model import registry

from tsn.util.distributed import get_device, get_local_rank
from tsn.model.layers.conv_helper import get_conv
from tsn.model.layers.pool_helper import get_pool
from tsn.model.layers.norm_helper import get_norm
from tsn.model.layers.act_helper import get_act

__all__ = ['ResNet3d', 'resnet3d_50', 'resnet3d_101',
           'resnet3d_152', ]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def _load_pretrained(arch, map_location=None):
    state_dict_2d = load_state_dict_from_url(model_urls[arch],
                                             progress=True,
                                             map_location=map_location)
    return state_dict_2d


def _resnet(arch, cfg, block_layer):
    pretrained2d = cfg.MODEL.BACKBONE.TORCHVISION_PRETRAINED
    state_dict_2d = None
    if pretrained2d:
        device = get_device(local_rank=get_local_rank())
        state_dict_2d = _load_pretrained(arch, map_location=device)

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
        # kernel_size of the first pooling layer
        pool1_kernel=cfg.MODEL.BACKBONE.POOL1_KERNEL,
        # stride of the first pooling layer
        pool1_stride=cfg.MODEL.BACKBONE.POOL1_STRIDE,
        # whether to use the second pooling layer
        with_pool2=cfg.MODEL.BACKBONE.WITH_POOL2,
        # kernel_size of the second pooling layer
        pool2_kernel=cfg.MODEL.BACKBONE.POOL2_KERNEL,
        # stride of the second pooling layer
        pool2_stride=cfg.MODEL.BACKBONE.POOL2_STRIDE,
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
        # whether to load a pretrained model
        state_dict_2d=state_dict_2d,
        # whether to apply partial BN
        partial_bn=cfg.MODEL.BACKBONE.PARTIAL_BN)
    return model


@registry.BACKBONE.register('R3D50')
def resnet3d_50(cfg):
    return _resnet("resnet50", cfg, Bottleneck3d)


@registry.BACKBONE.register('R3D101')
def resnet3d_101(cfg):
    return _resnet("resnet101", cfg, Bottleneck3d)


@registry.BACKBONE.register('R3D152')
def resnet3d_152(cfg):
    return _resnet("resnet152", cfg, Bottleneck3d)
