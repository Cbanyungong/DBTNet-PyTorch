import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


class SemanticGroupingLayer(nn.Module):

    def __init__(self, in_channels, out_channels, batch_size, groups, device):
        super(SemanticGroupingLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.groups = groups

        self.sg_conv = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels, out_channels, 1, bias=False)),
            ("bn", nn.BatchNorm2d(out_channels)),
            ("relu", nn.ReLU(inplace=True))
        ]))

        self.gt = torch.ones(self.groups).diag().to(device)
        self.gt = self.gt.reshape((1, 1, self.groups, self.groups))
        self.gt = self.gt.repeat((1, int((self.out_channels / self.groups) ** 2), 1, 1))
        self.gt = F.pixel_shuffle(self.gt, upscale_factor=int(self.out_channels / self.groups))
        self.gt = self.gt.reshape((1, self.out_channels ** 2))

        self.loss = 0

    def forward(self, x):

        # calculate act
        act = self.sg_conv(x)

        b, c, w, h = x.shape

        # loss
        tmp = act + 1e-3
        tmp = tmp.reshape((-1, w*h))
        tmp = F.instance_norm(tmp)
        tmp = tmp.reshape((-1, self.out_channels, w*h))
        tmp = tmp.permute(1, 0, 2).reshape(self.out_channels, -1)

        co_matrix = torch.matmul(tmp, tmp.t()).reshape((1, self.out_channels**2))
        co_matrix /= self.batch_size

        loss = torch.sum((co_matrix-self.gt)*(co_matrix-self.gt)*0.001, dim=1).repeat(self.batch_size)
        self.loss = loss/((self.out_channels/512.0)**2)
        return act


class GroupBilinearLayer(nn.Module):

    def __init__(self, groups, channels):
        super(GroupBilinearLayer, self).__init__()

        self.groups = groups
        self.channels = channels
        self.channels_per_group = int(self.channels/self.groups)

        self.fc = nn.Linear(channels, channels)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):

        b, c, w, h = x.shape

        tmp = x.permute(0, 2, 3, 1).reshape((-1, self.channels))
        tmp += self.fc(tmp)

        tmp = tmp.reshape(-1, self.groups, self.channels_per_group)
        tmp_t = tmp.permute(0, 2, 1)
        tmp = torch.tanh(torch.bmm(tmp_t, tmp)/32).reshape(-1, w, h, self.channels_per_group**2)
        tmp = F.interpolate(tmp, (h, c), mode="bilinear")
        tmp = tmp.permute(0, 3, 1, 2)

        out = x + self.bn(tmp)
        return out


class DBTNetBlock(nn.Module):

    def __init__(self, inplanes, planes, batch_size, device, groups=None, downsample=False, use_dbt=False):
        super(DBTNetBlock, self).__init__()

        self.block = nn.Sequential()
        expansion = 4
        if downsample:
            stride = 2
        else:
            stride = 1

        if use_dbt:
            self.sg_layer = SemanticGroupingLayer(inplanes, planes, batch_size, groups, device)
            self.gb_layer = GroupBilinearLayer(groups, planes)
            self.block.add_module("sg_layer", self.sg_layer)
            self.block.add_module("gb_layer", self.gb_layer)
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

    def __init__(self, layers, batch_size, device, classes=1000):
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
        self.features.add_module("block_1", self._make_layer(64, layers[0], batch_size, device, use_dbt=False))
        self.features.add_module("block_2", self._make_layer(128, layers[1], batch_size, device, use_dbt=False))
        self.features.add_module("block_3", self._make_layer(256, layers[2], batch_size, device, groups=16, use_dbt=True))
        self.features.add_module("block_4", self._make_layer(512, layers[3], batch_size, device, groups=16, use_dbt=True))

        # -------------
        # last DBT module
        # ------------
        self.last_sg_layer = SemanticGroupingLayer(self.inplanes, self.inplanes, batch_size, 32, device)
        self.last_gb_layer = GroupBilinearLayer(32, self.inplanes)
        self.features.add_module("last_sg_layer", self.last_sg_layer)
        self.features.add_module("last_gb_layer", self.last_gb_layer)

        # -------------
        # output
        # -------------
        # channels: inplanes -> classes
        self.features.add_module("output_11conv", nn.Conv2d(self.inplanes, classes, 1))
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, base_planes, blocks_num, batch_size, device, groups=None, use_dbt=False):
        layers = nn.Sequential()

        layers.add_module("dbtblock_1", DBTNetBlock(inplanes=self.inplanes, planes=base_planes, batch_size=batch_size,
                                                    device=device, groups=groups, downsample=True, use_dbt=use_dbt))
        self.inplanes = base_planes * self.expansion

        for i in range(blocks_num-1):
            layers.add_module("dbtblock_{}".format(i+2), DBTNetBlock(inplanes=self.inplanes, planes=base_planes,
                                                                     batch_size=batch_size, device=device,
                                                                     groups=groups, use_dbt=use_dbt))

        return layers

    def forward(self, x):
        x = self.features(x)
        x = self.avgpooling(x)
        x = x.view(x.shape[0], -1)
        return x


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dbt_net = DBTNet([3, 4, 6, 3], 48, device).to(device)
    test_noise = torch.randn((12, 3, 448, 448)).to(device)
    output = dbt_net(test_noise)
    print(output.shape)
