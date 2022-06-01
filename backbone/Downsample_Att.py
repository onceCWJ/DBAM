import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class SCConvBlock1(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConvBlock1, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(inplanes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(inplanes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )


    def forward(self, x):
        identity = x
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

class SCConvBlock2(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConvBlock2, self).__init__()
        self.k1 = nn.Sequential(
            nn.Conv2d(
                inplanes, planes, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=groups, bias=False),
            norm_layer(planes)
        )

    def forward(self, x):
        identity = x
        out = self.k1(identity) # k4

        return out

class Downsample_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding=1, dilation=1, groups=1, pooling_r=3, norm_layer=nn.BatchNorm2d):
        super(Downsample_Attention, self).__init__()
        self.Block1 = SCConvBlock1(in_channels, out_channels, stride, padding, dilation, groups, pooling_r, norm_layer)
        self.Block2 = SCConvBlock2(in_channels, out_channels, stride, padding, dilation, groups, pooling_r, norm_layer)

    def forward(self, x):
        y, z = torch.chunk(x, 2, dim=1)
        out_1 = self.Block1(y)
        out_2 = self.Block2(z)
        out = torch.cat([out_1, out_2], dim=1)

        return (out)

