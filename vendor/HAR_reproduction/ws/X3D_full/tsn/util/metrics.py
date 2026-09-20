# -*- coding: utf-8 -*-

"""
@date: 2020/4/27 8:25 PM
@file: metrics.py
@author: zj
@description: 
"""

import torch
from thop import profile

from torchvision.models import AlexNet


def compute_num_flops(model):
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input,), verbose=False)
    # print(macs, params)

    GFlops = macs * 2.0 / pow(10, 9)
    params_size = params * 4.0 / 1024 / 1024
    return GFlops, params_size


def topk_accuracy(output, target, topk=(1,)):
    """
    Computes top-K accuracy. N is the number of samples, C is the number of classes
    :param output: tensor of shape [N, C]; each row contains the C class probabilities computed for one sample
    :param target: tensor of shape [N]; each element is the ground-truth class index
    :param topk: tuple of top-k accuracies to compute
    :return: list
    """
    assert len(output.shape) == 2 and output.shape[0] == target.shape[0]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
