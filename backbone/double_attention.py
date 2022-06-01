import torch.nn as nn
import torch
from torch.nn import functional as F
from .feature_pyramid_network import LastLevelMaxPool, FeaturePyramidNetwork
import numpy as np


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * (scale)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        scale = torch.sigmoid(x)
        x = (scale) * residual

        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# 尝试在下采样过程中加入FPN的结构
class downsample_att(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=4):
        super(downsample_att, self).__init__()
        self.compress = ChannelPool()
        self.activa = nn.ReLU()
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(2, 1, stride=1, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, stride=1, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, stride=1, kernel_size=3, dilation=3, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, kernel_size=1, stride=stride)
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))
        
        self.pool = ChannelPool()
        # self.channel_att = ChannelGate(out_channels)
    # 可以先对于整体特征图进行降维，降到只有三个channel，分别记录：最大值、平均值、方差
    # 随后进行dilation操作，得到最后的数值，再接上一些其他的注意力机制，得到最终结果
    # 下采样注意力机制设计的最大问题在于一旦进行下采样就不是原来的特征图了，这样出现了问题
    # 如何做到注意力机制也能做到伸缩性并且与下采样操作相适配是个问题
    # 能不能一边进行下采样操作，一边进行注意力机制的计算，这样注意力机制也补足了下采样操作中丢失的信息
    def forward(self, x):
        x_out = self.pool(x)
        scale = torch.sigmoid(self.dilated_conv(x_out))
        x_out = self.conv(x) * scale
        return x_out


class DBAM(nn.Module):
    def __init__(self, channels, out_channels=None, no_spatial=True):
        super(DBAM, self).__init__()
        self.Channel_Att = ChannelGate(channels)
        self.Spatial_Att = STNCA(channels)

    def forward(self, x):
        x_out = self.Channel_Att(x)
        x_out = self.Spatial_Att(x_out)
        return x_out


class SimAM(nn.Module):
    def __init__(self, lambdax=1e-5):
        super(SimAM, self).__init__()
        self.lambdax = 1e-5
        self.activate = torch.sigmoid

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1
        x_minus = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        e_inv = x_minus / (4 * (x_minus.sum(dim=[2, 3], keepdim=True) / n + self.lambdax)) + 0.5

        return x * self.activate(e_inv)


class STNCA(nn.Module):
    def __init__(self, in_channels, device='cuda', drop_prob=0.8):
        super(STNCA, self).__init__()
        self.drop_prob = drop_prob
        self.activa = nn.Tanh()
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(in_features = in_channels * 6 * 6, out_features = 20),
            nn.Tanh(),
            nn.Dropout(self.drop_prob),
            nn.Linear(in_features = 20, out_features=6),
            nn.Tanh()
        )

        bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))
        self.pool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        nn.init.constant_(self.fc[3].weight, 0)
        self.fc[3].bias.data.copy_(bias)

    def forward(self, img):
        '''
        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''

        out_img = self.pool(img)
        batch_size = img.size(0)

        theta = self.activa(self.fc(out_img.view(batch_size, -1)).view(batch_size, 2, 3))

        grid = F.affine_grid(theta, torch.Size((batch_size, img.shape[1], img.shape[2], img.shape[3])))
        img_transform = F.grid_sample(img, grid, align_corners=True)

        return img_transform