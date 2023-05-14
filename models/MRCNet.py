"""
Original implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018
Jinwei Gu and Zhile Ren
Modified version (CMRNet) by Daniele Cattaneo
Modified version (LCCNet) by Xudong Lv
Modified version (MRCNet) by Hao Wang
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
#from models.CMRNet.modules.attention import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import argparse
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# from .networks.submodules import *
# from .networks.correlation_package.correlation import Correlation
#from models.correlation_package.correlation import Correlation
from spatial_correlation_sampler import SpatialCorrelationSampler


# __all__ = [
#     'calib_net'
# ]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out




class SEBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.leakyRELU = nn.LeakyReLU(0.1)
        # self.attention = SCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = SElayer(planes * self.expansion, ratio=reduction)
        # self.attention = ECAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = SElayer_conv(planes * self.expansion, ratio=reduction)
        # self.attention = SCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = ModifiedSCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = DPCSAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = PAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = CAlayer(planes * self.expansion, ratio=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRELU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyRELU(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leakyRELU(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def myconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=1, bias=True),
        nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        # self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.maxpool(self.features[-1]))
        self.features.append(self.encoder.layer1(self.features[-1]))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class MRCNet(nn.Module):
    """
    Based on the PWC-DC net. add resnet encoder, dilation convolution and densenet connections
    """

    def __init__(self, image_size, use_feat_from=1, md=4, use_reflectance=False, dropout=0.0,
                 Action_Func='leakyrelu', attention=False, res_num=18):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(MRCNet, self).__init__()
        self.toplayer = nn.Conv2d(512, 64, 1, 1, 0)

        self.smooth1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.smooth2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.smooth3 = nn.Conv2d(64, 64, 3, 1, 1)

        self.latlayer1 = nn.Conv2d(256, 64, 1, 1, 0)
        self.latlayer2 = nn.Conv2d( 128, 64, 1, 1, 0)
        self.latlayer3 = nn.Conv2d( 64, 64, 1, 1, 0)

        input_lidar = 1
        self.res_num = res_num
        self.use_feat_from = use_feat_from
        if use_reflectance:
            input_lidar = 2

        # original resnet
        self.pretrained_encoder = False
        self.net_encoder = ResnetEncoder(num_layers=self.res_num, pretrained=True, num_input_images=1)

        # resnet with leakyRELU
        self.Action_Func = Action_Func
        self.attention = attention
        self.inplanes = 64
        if self.res_num == 50:
            layers = [3, 4, 6, 3]
            add_list = [1024, 512, 256, 64]
        elif self.res_num == 18:
            layers = [2, 2, 2, 2]
            add_list = [256, 128, 64, 64]

        if self.attention:
            block = SEBottleneck
        else:
            if self.res_num == 50:
                block = Bottleneck
            elif self.res_num == 18:
                block = BasicBlock


        # rgb_image
        self.conv1_rgb = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.elu_rgb = nn.ELU()
        self.leakyRELU_rgb = nn.LeakyReLU(0.1)
        self.maxpool_rgb = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_rgb = self._make_layer(block, 64, layers[0])
        self.layer2_rgb = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_rgb = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_rgb = self._make_layer(block, 512, layers[3], stride=2)

        # lidar_image
        self.inplanes = 64
        self.conv1_lidar = nn.Conv2d(input_lidar, 64, kernel_size=7, stride=2, padding=3)
        self.elu_lidar = nn.ELU()
        self.leakyRELU_lidar = nn.LeakyReLU(0.1)
        self.maxpool_lidar = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_lidar = self._make_layer(block, 64, layers[0])
        self.layer2_lidar = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_lidar = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_lidar = self._make_layer(block, 512, layers[3], stride=2)

        self.corr = SpatialCorrelationSampler(
    kernel_size=1,
    patch_size=9,
    stride=1,
    padding=0,
    dilation_patch=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd
        self.conv6_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)
        self.conv9_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv9_1 = myconv(128, 64, kernel_size=3, stride=1)
        self.conv9_2 = myconv(64, 32, kernel_size=3, stride=1)
        self.conv7_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv7_1 = myconv(128, 64, kernel_size=3, stride=1)
        self.conv7_2 = myconv(64, 16, kernel_size=3, stride=1)
        self.conv8_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv8_1 = myconv(128, 64, kernel_size=3, stride=1)
        self.conv8_2 = myconv(64, 8, kernel_size=3, stride=1)


        fc_size = od + dd[4]
        downsample = 128 // (2**use_feat_from)
        if image_size[0] % downsample == 0:
            fc_size *= image_size[0] // downsample
        else:
            fc_size *= (image_size[0] // downsample)+1
        if image_size[1] % downsample == 0:
            fc_size *= image_size[1] // downsample
        else:
            fc_size *= (image_size[1] // downsample)+1
        # self.fc1 = nn.Linear(10368 , 512)
        # self.fc2 = nn.Linear(4*10368 , 512)
        # self.fc3 = nn.Linear(16*10368 , 512)
        # self.fc4 = nn.Linear(64*10368 , 512)

        self.fc1_trasl = nn.Linear(271072, 64)
        self.fc1_rot = nn.Linear(271072, 64)
        self.fc115_trasl=nn.Linear(16384, 128)
        self.fc114_trasl=nn.Linear(4*8192, 64)
        self.fc113_trasl=nn.Linear(4*16384, 32)
        self.fc2_trasl = nn.Linear(64, 3)
        self.fc2_rot = nn.Linear(64, 4)

        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    # def _make_layer(self, block, planes, blocks, stride=1):
    #     layers = []
    #     layers.append(block(self.inplanes, planes, 1))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks - 1):
    #         layers.append(block(self.inplanes, planes, 1))
    #     layers.append(block(self.inplanes, planes, 2))
    #
    #     return nn.Sequential(*layers)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def _upsample_add(self, x, y):
        _,_,H,W = y.shape
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        # mask[mask<0.9999] = 0.0
        # mask[mask>0] = 1.0
        mask = torch.floor(torch.clamp(mask, 0, 1))

        return output * mask

    def forward(self, rgb, lidar):

        #encoder
        if self.pretrained_encoder:
            # rgb_image
            features1 = self.net_encoder(rgb)
            c12 = features1[0]  # 2
            c13 = features1[2]  # 4
            c14 = features1[3]  # 8
            c15 = features1[4]  # 16
            c16 = features1[5]  # 32
            # lidar_image
            x2 = self.conv1_lidar(lidar)
            if self.Action_Func == 'leakyrelu':
                c22 = self.leakyRELU_lidar(x2)  # 2
            elif self.Action_Func == 'elu':
                c22 = self.elu_lidar(x2)  # 2
            c23 = self.layer1_lidar(self.maxpool_lidar(c22))  # 4
            c24 = self.layer2_lidar(c23)  # 8
            c25 = self.layer3_lidar(c24)  # 16
            c26 = self.layer4_lidar(c25)  # 32

        else:
            x1 = self.conv1_rgb(rgb)
            x2 = self.conv1_lidar(lidar)
            if self.Action_Func == 'leakyrelu':
                c12 = self.leakyRELU_rgb(x1)  # 2
                c22 = self.leakyRELU_lidar(x2)  # 2
            elif self.Action_Func == 'elu':
                c12 = self.elu_rgb(x1)  # 2
                c22 = self.elu_lidar(x2)  # 2
            c13 = self.layer1_rgb(c12)  # 4
            c23 = self.layer1_lidar(c22)  # 4
            c14 = self.layer2_rgb(c13)  # 8
            c24 = self.layer2_lidar(c23)  # 8
            c15 = self.layer3_rgb(c14)  # 16
            c25 = self.layer3_lidar(c24)  # 16
            c16 = self.layer4_rgb(c15)  # 32
            c26 = self.layer4_lidar(c25)  # 32
        p5 = self.toplayer(c26)
        p4 = self._upsample_add(p5, self.latlayer1(c25))
        p3 = self._upsample_add(p4, self.latlayer2(c24))


        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)

        p15 = self.toplayer(c16)
        p14 = self._upsample_add(p15, self.latlayer1(c15))

        p13 = self._upsample_add(p14, self.latlayer2(c14))

        p14 = self.smooth1(p14)
        p13 = self.smooth2(p13)

        corr5 = self.corr(p5, p15)
        b,ph,pw,h,w=corr5.size()
        corr5=corr5.view(b,ph*pw,h,w)/c16.size(1)
        x115=self.leakyRELU(corr5)
        x115=self.conv9_0(x115)
        x115=self.conv9_1(x115)
        x115=self.conv9_2(x115)

        x115 = x115.view(x115.shape[0], -1)
        x115 = self.dropout(x115)
        x115 = self.leakyRELU(x115)
        x115=self.leakyRELU(self.fc115_trasl(x115))

        corr4 = self.corr(p4, p14)
        b,ph,pw,h,w=corr4.size()
        corr4=corr4.view(b,ph*pw,h,w)/c16.size(1)
        x114=self.leakyRELU(corr4)
        x114=self.conv7_0(x114)
        x114=self.conv7_1(x114)
        x114=self.conv7_2(x114)
        x114 = x114.view(x114.shape[0], -1)
        x114 = self.dropout(x114)
        x114 = self.leakyRELU(x114)
        x114=self.leakyRELU(self.fc114_trasl(x114))
        corr3 = self.corr(p3, p13)
        b,ph,pw,h,w=corr3.size()
        corr3=corr3.view(b,ph*pw,h,w)/c16.size(1)
        x113=self.leakyRELU(corr3)
        x113=self.conv8_0(x113)
        x113=self.conv8_1(x113)
        x113=self.conv8_2(x113)
        x113 = x113.view(x113.shape[0], -1)
        x113 = self.dropout(x113)
        x113 = self.leakyRELU(x113)
        x113=self.leakyRELU(self.fc113_trasl(x113))

        corr6 = self.corr(c16, c26)
        b,ph,pw,h,w=corr6.size()
        corr6=corr6.view(b,ph*pw,h,w)/c16.size(1)
        corr6 = self.leakyRELU(corr6)
        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)

        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.leakyRELU(x)
        x = torch.cat((x113, x), 1)
        x = torch.cat((x114, x), 1)
        x = torch.cat((x115, x), 1)
        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))

        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)

        return transl, rot
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.elu = nn.ELU()
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRELU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyRELU(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leakyRELU(out)

        return out

