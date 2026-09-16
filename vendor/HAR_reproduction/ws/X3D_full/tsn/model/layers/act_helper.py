# -*- coding: utf-8 -*-

"""
@date: 2020/11/3 10:22 AM
@file: act_helper.py
@author: zj
@description: 
"""

import torch.nn as nn


def get_act(type):
    if type == 'ReLU':
        return nn.ReLU
    else:
        raise ValueError(f'{type} does not exists')
