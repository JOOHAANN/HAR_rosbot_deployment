# -*- coding: utf-8 -*-

"""
@date: 2020/11/2 7:24 PM
@file: resnet3d.py
@author: zj
@description: 
"""

import torch.nn as nn

from .bottleneck3d import Bottleneck3d
from .basic_block3d import BasicBlock3d


class ResNet3d(nn.Module):

    def __init__(self,
                 # number of input channels
                 in_channels=3,
                 # number of stem channels
                 base_channel=64,
                 # kernel_size of the first conv layer
                 conv1_kernel=(1, 7, 7),
                 # stride of the first conv layer
                 conv1_stride=(2, 2, 2),
                 # zero padding of the first conv layer
                 conv1_padding=(0, 3, 3),
                 # whether to use the first pooling layer
                 with_pool1=True,
                 # kernel_size of the first pooling layer
                 pool1_kernel=(3, 3, 3),
                 # stride of the first pooling layer
                 pool1_stride=(2, 2, 2),
                 # whether to use the second pooling layer
                 with_pool2=True,
                 # kernel_size of the second pooling layer
                 pool2_kernel=(3, 1, 1),
                 # stride of the second pooling layer
                 pool2_stride=(2, 1, 1),
                 # number of blocks per stage, e.g. R50
                 stage_blocks=(3, 4, 6, 3),
                 # output channels of the first conv layer in each stage
                 res_planes=None,
                 # expansion factor, e.g. Bottleneck
                 expansion=4,
                 # spatial stride
                 spatial_strides=(1, 2, 2, 2),
                 # whether to inflate
                 inflates=(0, 0, 0, 0),
                 # inflation type
                 inflate_style='3x1x1',
                 # conv layer type
                 conv_layer=None,
                 # pooling layer type
                 pool_layer=None,
                 # norm layer type
                 norm_layer=None,
                 # activation layer type
                 act_layer=None,
                 # block type
                 block_layer=None,
                 # whether to zero-init the residual branch
                 zero_init_residual=True,
                 # whether to load a pretrained model
                 state_dict_2d=None,
                 # whether to apply partial BN
                 partial_bn=False
                 ):
        super(ResNet3d, self).__init__()
        assert len(stage_blocks) == len(spatial_strides) == len(inflates)

        if conv_layer is None:
            conv_layer = nn.Conv3d
        if pool_layer is None:
            pool_layer = nn.MaxPool3d
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if act_layer is None:
            act_layer = nn.ReLU
        if block_layer is None:
            block_layer = Bottleneck3d

        # number of input channels
        self.in_channels = in_channels
        # number of stem channels
        self.base_channel = base_channel
        # kernel_size of the first conv layer
        self.conv1_kernel = conv1_kernel
        # stride of the first conv layer
        self.conv1_stride = conv1_stride
        # whether to use the first pooling layer
        self.with_pool1 = with_pool1
        # kernel_size of the first pooling layer
        self.pool1_kernel = pool1_kernel
        # stride of the first pooling layer
        self.pool1_stride = pool1_stride
        # zero padding of the first conv layer
        self.conv1_padding = conv1_padding
        # whether to use the second pooling layer
        self.with_pool2 = with_pool2
        # kernel_size of the second pooling layer
        self.pool2_kernel = pool2_kernel
        # stride of the second pooling layer
        self.pool2_stride = pool2_stride
        # number of blocks per stage, e.g. R50
        self.stage_blocks = stage_blocks
        # output channels of the first conv layer in each stage
        self.res_planes = res_planes
        # expansion factor, e.g. Bottleneck
        self.expansion = expansion
        # spatial stride
        self.spatial_strides = spatial_strides
        # whether to inflate
        self.inflates = inflates
        # inflation type
        self.inflate_style = inflate_style
        # conv layer type
        self.conv_layer = conv_layer
        # pooling layer type
        self.pool_layer = pool_layer
        # norm layer type
        self.norm_layer = norm_layer
        # activation layer type
        self.act_layer = act_layer
        # block type
        self.block_layer = block_layer
        # whether to zero-init the residual branch
        self.zero_init_residual = zero_init_residual
        # whether to load a pretrained model
        self.state_dict_2d = state_dict_2d
        # whether to apply partial BN
        self.partial_bn = partial_bn

        self._make_stem_layer()

        self.res_layers = list()
        self.inplanes = self.base_channel
        res_planes = [self.base_channel,
                      self.base_channel * 2,
                      self.base_channel * 2 * 2,
                      self.base_channel * 2 * 2 * 2] if self.res_planes is None else self.res_planes
        for i in range(len(self.stage_blocks)):
            res_layer = self._make_res_layer(self.inplanes,
                                             res_planes[i],
                                             self.conv_layer,
                                             self.norm_layer,
                                             self.act_layer,
                                             self.block_layer,
                                             self.stage_blocks[i],
                                             self.expansion,
                                             self.spatial_strides[i],
                                             self.inflates[i],
                                             self.inflate_style)
            self.inplanes = int(res_planes[i] * self.expansion)

            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._init_weights(self.zero_init_residual,
                           self.state_dict_2d)

    def _make_stem_layer(self):
        self.conv1 = self.conv_layer(self.in_channels,
                                     self.base_channel,
                                     kernel_size=self.conv1_kernel,
                                     stride=self.conv1_stride,
                                     padding=self.conv1_padding,
                                     bias=False)
        self.bn1 = self.norm_layer(self.base_channel)
        self.act = self.act_layer(inplace=True)

        if self.with_pool1:
            self.pool1 = self.pool_layer(kernel_size=self.pool1_kernel,
                                         stride=self.pool1_stride,
                                         ceil_mode=True)
        if self.with_pool2:
            self.pool2 = self.pool_layer(kernel_size=self.pool2_kernel,
                                         stride=self.pool2_stride,
                                         ceil_mode=True)

    def _make_res_layer(self,
                        # number of input channels
                        inplanes,
                        # output channels of the first conv layer
                        planes,
                        # conv layer type
                        conv_layer,
                        # norm layer type
                        norm_layer,
                        # activation layer type
                        act_layer,
                        # block type
                        block_layer,
                        # number of blocks
                        block_num,
                        # expansion factor
                        expansion,
                        # spatial stride; whether the first block downsamples
                        spatial_stride,
                        # whether to inflate the temporal dimension
                        inflate,
                        # inflation type
                        inflate_style
                        ):
        inflate = inflate if not isinstance(inflate, int) else (inflate,) * block_num
        assert len(inflate) == block_num

        layers = []
        for i in range(block_num):
            layers.append(
                block_layer(
                    inplanes,
                    planes,
                    spatial_stride=1 if i != 0 else spatial_stride,
                    inflate=(inflate[0] == 1),
                    inflate_style=inflate_style,
                    expansion=expansion,
                    conv_layer=conv_layer,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                ))
            inplanes = int(planes * expansion)
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual, state_dict_2d):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                batchnorm_weight = 1.0
                nn.init.constant_(m.weight, batchnorm_weight)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3d):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock3d):
                    nn.init.constant_(m.bn2.weight, 0)

        if state_dict_2d:

            def _inflate_conv_params(conv3d, state_dict_2d, module_name_2d,
                                     inflated_param_names):
                """Inflate a conv module from 2d to 3d.

                Args:
                    conv3d (nn.Module): The destination conv3d module.
                    state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
                    module_name_2d (str): The name of corresponding conv module in the
                        2d model.
                    inflated_param_names (list[str]): List of parameters that have been
                        inflated.
                """
                weight_2d_name = module_name_2d + '.weight'

                conv2d_weight = state_dict_2d[weight_2d_name]
                kernel_t = conv3d.weight.data.shape[2]

                new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
                    conv3d.weight) / kernel_t
                conv3d.weight.data.copy_(new_weight)
                inflated_param_names.append(weight_2d_name)

                if getattr(conv3d, 'bias') is not None:
                    bias_2d_name = module_name_2d + '.bias'
                    conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
                    inflated_param_names.append(bias_2d_name)

            def _inflate_bn_params(bn3d, state_dict_2d, module_name_2d,
                                   inflated_param_names):
                """Inflate a norm module from 2d to 3d.

                Args:
                    bn3d (nn.Module): The destination bn3d module.
                    state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
                    module_name_2d (str): The name of corresponding bn module in the
                        2d model.
                    inflated_param_names (list[str]): List of parameters that have been
                        inflated.
                """
                for param_name, param in bn3d.named_parameters():
                    param_2d_name = f'{module_name_2d}.{param_name}'
                    param_2d = state_dict_2d[param_2d_name]
                    param.data.copy_(param_2d)
                    inflated_param_names.append(param_2d_name)

                for param_name, param in bn3d.named_buffers():
                    param_2d_name = f'{module_name_2d}.{param_name}'
                    # some buffers like num_batches_tracked may not exist in old
                    # checkpoints
                    if param_2d_name in state_dict_2d:
                        param_2d = state_dict_2d[param_2d_name]
                        param.data.copy_(param_2d)
                        inflated_param_names.append(param_2d_name)

            inflated_param_names = []
            keys = state_dict_2d.keys()
            for name, module in self.named_modules():
                if isinstance(module, nn.Conv3d):
                    _inflate_conv_params(module, state_dict_2d, name, inflated_param_names)
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    _inflate_bn_params(module, state_dict_2d, name, inflated_param_names)

            # check if any parameters in the 2d checkpoint are not loaded
            remaining_names = set(keys) - set(inflated_param_names)
            if remaining_names:
                print(f'These parameters in the 2d checkpoint are not loaded: {sorted(remaining_names)}')

    def freezing_bn(self):
        count = 0
        for m in self.modules():
            if isinstance(m, type(self.norm_layer)):
                count += 1
                if count == 1:
                    continue

                m.eval()
                # shutdown update in frozen mode
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def train(self, mode: bool = True):
        super(ResNet3d, self).train(mode=mode)

        if mode and self.partial_bn:
            self.freezing_bn()

        return self

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        assert len(x.shape) == 5

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        if self.with_pool1:
            x = self.pool1(x)

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)

        return x
