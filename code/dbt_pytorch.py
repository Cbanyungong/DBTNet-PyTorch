import os
import numpy as np
from collections import OrderedDict

import torch
from torch import nn

from torchvision.models import resnet101


class DBTNetBlock(nn.Module):

    def __init__(self, inplanes, planes, batch_size, downsample=False, use_dbt=False):
        super(DBTNetBlock, self).__init__()

        self.block = nn.Sequential()
        expansion = 4
        if downsample:
            stride = 2
        else:
            stride = 1

        if use_dbt:
            self.block.add_module("conv_1", nn.Conv2d(inplanes, planes, 1, bias=False))
            self.block.add_module("bn_1", nn.BatchNorm2d(planes))
            self.block.add_module("relu_1", nn.ReLU(inplace=True))
        else:
            self.block.add_module("conv_1", nn.Conv2d(inplanes, planes, 1, bias=False))
            self.block.add_module("bn_1", nn.BatchNorm2d(planes))
            self.block.add_module("relu_1", nn.ReLU(inplace=True))

        self.block.add_module("conv 2", nn.Conv2d(planes, planes, 3, stride, 1, bias=False))
        self.block.add_module("bn_2", nn.BatchNorm2d(planes))
        self.block.add_module("relu_2", nn.ReLU(inplace=True))

        self.block.add_module("conv_3", nn.Conv2d(planes, planes*expansion, 1, bias=False))

        self.downsample = nn.Sequential()
        if downsample:
            self.downsample.add_module("conv_downsample", nn.Conv2d(inplanes, planes*expansion,
                                                                    1, 2, bias=False))
            self.downsample.add_module("bn_downsample", nn.BatchNorm2d(planes*expansion))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        x = self.block(x)

        output = self.relu(x+identity)
        return output


class DBTNet(nn.Module):

    def __init__(self, layers, batch_size, classes=1000):
        super(DBTNet, self).__init__()
        self.inplanes = 64
        self.features = nn.Sequential()
        self.expansion = 4

        # -------------
        # input conv
        # -------------
        # BN-Conv-BN-ReLU-Max pooling
        self.input_ = nn.Sequential(OrderedDict([
            ("input_conv", nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)),
            ("input_bn", nn.BatchNorm2d(self.inplanes)),
            ("input_activate", nn.ReLU(inplace=True)),
            ("input_maxpooling", nn.MaxPool2d(3, 2, 1))
        ]))
        self.features.add_module("input_features", self.input_)

        # -------------
        # resnet-dbt block
        # -------------
        self.features.add_module("block_1", self._make_layer(64, layers[0], batch_size, use_dbt=False))
        self.features.add_module("block_2", self._make_layer(128, layers[1], batch_size, use_dbt=False))
        self.features.add_module("block_3", self._make_layer(256, layers[2], batch_size, use_dbt=True))
        self.features.add_module("block_4", self._make_layer(512, layers[3], batch_size, use_dbt=True))

        # -------------
        # output
        # -------------
        # channels: inplanes -> classes
        self.features.add_module("output_11conv", nn.Conv2d(self.inplanes, classes, 1))
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, base_planes, blocks_num, batch_size, use_dbt=False):
        layers = nn.Sequential()

        layers.add_module("dbtblock_1", DBTNetBlock(inplanes=self.inplanes, planes=base_planes, batch_size=batch_size,
                                                    downsample=True, use_dbt=use_dbt))
        self.inplanes = base_planes * self.expansion

        for i in range(blocks_num-1):
            layers.add_module("dbtblock_{}".format(i+2), DBTNetBlock(inplanes=self.inplanes, planes=base_planes,
                                                                     batch_size=batch_size))

        return layers

    def forward(self, x):
        x = self.features(x)
        x = self.avgpooling(x)
        x = x.view(x.shape[0], -1)
        return x


if __name__ == "__main__":
    dbt_net = DBTNet([3, 4, 6, 3], 48)
    test_noise = torch.randn((48, 3, 448, 448))
    output = dbt_net(test_noise)
    print(output.shape)
