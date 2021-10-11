from __future__ import print_function

from collections import OrderedDict
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

from .utils import compute_same_padding2d


class Conv2d_dilated(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 NL='relu', same_padding=False, dilation=1,
                 bn=False, bias=True, groups=1
                 ):
        super(Conv2d_dilated, self).__init__()
        self.conv = _Conv2d_dilated(in_channels, out_channels, kernel_size,
                                    stride, dilation=dilation, groups=groups,
                                    bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0,
                                 affine=True) if bn else None
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        elif NL == 'tanh':
            self.relu = nn.Tanh()
        elif NL == 'lrelu':
            self.relu = nn.LeakyReLU(inplace=True)
        elif NL == 'sigmoid':
            self.relu = nn.Sigmoid()
        else:
            self.relu = None

    def forward(self, x, dilation=None):
        x = self.conv(x, dilation)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class _Conv2d_dilated(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True
                 ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        super(_Conv2d_dilated, self).__init__(
            in_channels, out_channels, kernel_size, stride, _pair(0), dilation,
            False, _pair(0), groups, bias, 'zeros')

    def forward(self, input, dilation=None):
        input_shape = list(input.size())
        dilation_rate = self.dilation if dilation is None else _pair(dilation)
        padding, pad_input = compute_same_padding2d(input_shape,
                                                    kernel_size=self.kernel_size,
                                                    strides=self.stride,
                                                    dilation=dilation_rate)

        if pad_input[0] == 1 or pad_input[1] == 1:
            input = F.pad(input, [0, int(pad_input[0]), 0, int(pad_input[1])])
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        (padding[0] // 2, padding[1] // 2), dilation_rate,
                        self.groups)
        # https://github.com/pytorch/pytorch/issues/3867


class SequentialEndpoints(nn.Module):

    def __init__(self, layers, endpoints=None):
        super(SequentialEndpoints, self).__init__()
        assert isinstance(layers, OrderedDict)
        for key, module in layers.items():
            self.add_module(key, module)
        if endpoints is not None:
            self.Endpoints = namedtuple('Endpoints', endpoints.values(),
                                        verbose=True)
            self.endpoints = endpoints

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def sub_forward(self, startpoint, endpoint):
        def forward(input):
            flag = False
            output = None
            for key, module in self._modules.items():
                if startpoint == endpoint:
                    output = input
                    if key == startpoint:
                        output = module(output)
                        return output
                elif flag or key == startpoint:
                    if key == startpoint:
                        output = input
                    flag = True
                    output = module(output)
                    if key == endpoint:
                        return output
            return output

        return forward

    def forward(self, input, require_endpoints=False):
        if require_endpoints:
            endpoints = self.Endpoints([None] * len(self.endpoints.keys()))
        for key, module in self._modules.items():
            input = module(input)
            if require_endpoints and key in self.endpoints.keys():
                setattr(endpoints, self.endpoints[key], input)
        if require_endpoints:
            return input, endpoints
        else:
            return input


def load_net(fname, net, skip=False, prefix=''):
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    import h5py
    with h5py.File(fname, mode='r') as h5f:
        for k, v in net.state_dict().items():
            if skip:
                if 'relu' in k:
                    v.copy_(torch.from_numpy(np.zeros((1,))))
                    continue
            if 'loss' in k:
                # print(k)
                continue
            assert (prefix + k) in h5f.keys(), "key: {} size: {}".format(k,
                                                                         v.size())
            param = torch.from_numpy(np.asarray(h5f[(prefix + k)]))
            assert v.size() == param.size(), "{}: h5~{}-need~{}".format(k,
                                                                        param.size(),
                                                                        v.size())
            v.copy_(param)
