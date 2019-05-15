#coding=utf-8
import torch
import torch.nn as nn
from torch.nn import Upsample
from torch.autograd import Variable
import torch.nn.functional as F


# the implement of Hourglass Structure is right
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(inplanes, planes, padding=0,
                               kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, padding=1,
                               kernel_size=3, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, padding=0,
                               kernel_size=1, stride=1, bias=False)
        if stride != 1 or inplanes != planes * self.expansion:
            downsample = nn.Conv2d(inplanes, planes * self.expansion, padding=0,
                                   kernel_size=1, stride=stride, bias=False)
        self.downsample = downsample

        for m in self.modules():
            if m.__class__.__name__ in ['Conv2d']:
                nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            out = self.conv1(x)
            out = self.bn2(out)
            out = self.relu2(out)
            out = self.conv2(out)
            out = self.bn3(out)
            out = self.relu3(out)
            out = self.conv3(out)
        else:
            out = self.bn1(x)
            out = self.relu1(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu2(out)
            out = self.conv2(out)
            out = self.bn3(out)
            out = self.relu3(out)
            out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out

class Hourglass(nn.Module):

    def __init__(self, block=Bottleneck, num_blocks=1, planes=128//4, depth=4):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.hg = self._make_hourglass(block, num_blocks, planes,depth)

        for m in self.modules():
            if m.__class__.__name__ in ['Conv2d']:
                nn.init.kaiming_uniform_(m.weight.data)

    @staticmethod
    def _make_residual(block, num_blocks, planes):
        layers = []
        for index in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hourglass(self, block, num_blocks, planes, depth):
        hourglass = []
        for index in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if index == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hourglass.append(nn.ModuleList(res))
        return nn.ModuleList(hourglass)

    def _hourglass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = self.maxpool(x)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hourglass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2, mode='bilinear', align_corners=True)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hourglass_forward(self.depth, x)

# class HourGlass(nn.Module):
#     """不改变特征图的高宽"""
#     def __init__(self,n=4,f=128):
#         """
#         :param n: hourglass模块的层级数目
#         :param f: hourglass模块中的特征图数量
#         :return:
#         """
#         super(HourGlass,self).__init__()
#         self._n = n
#         self._f = f
#         self._init_layers(self._n,self._f)
#
#     def _init_layers(self,n,f):
#         # 上分支
#         setattr(self,'res'+str(n)+'_1',Residual(f,f))
#         # 下分支
#         setattr(self,'pool'+str(n)+'_1',nn.MaxPool2d(2,2))
#         setattr(self,'res'+str(n)+'_2',Residual(f,f))
#         if n > 1:
#             self._init_layers(n-1,f)
#         else:
#             self.res_center = Residual(f,f)
#         setattr(self,'res'+str(n)+'_3',Residual(f,f))
#         setattr(self,'unsample'+str(n),Upsample(scale_factor=2))
#
#
#     def _forward(self,x,n,f):
#         # 上分支
#         up1 = x
#         up1 = eval('self.res'+str(n)+'_1')(up1)
#         # 下分支
#         low1 = eval('self.pool'+str(n)+'_1')(x)
#         low1 = eval('self.res'+str(n)+'_2')(low1)
#         if n > 1:
#             low2 = self._forward(low1,n-1,f)
#         else:
#             low2 = self.res_center(low1)
#         low3 = low2
#         low3 = eval('self.'+'res'+str(n)+'_3')(low3)
#         up2 = eval('self.'+'unsample'+str(n)).forward(low3)
#
#         return up1+up2
#
#     def forward(self,x):
#         return self._forward(x,self._n,self._f)

class Residual(nn.Module):
    """
    残差模块，并不改变特征图的宽高
    """
    def __init__(self,ins,outs):
        super(Residual,self).__init__()
        # 卷积模块
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins,int(outs/2),1),
            nn.BatchNorm2d(int(outs/2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs // 2, int(outs / 2), 3, 1, 1),
            #nn.Conv2d(outs/2,int(outs/2),3,1,1),
            nn.BatchNorm2d(int(outs/2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outs/2),outs,1)
        )
        # 跳层
        if ins != outs:
            self.skipConv = nn.Conv2d(ins,outs,1)
        self.ins = ins
        self.outs = outs
    def forward(self,x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x

# 最后一层相当于把转换为对应数量的feature map
class Lin(nn.Module):
    # input feature map num = 128, output feature map = 11
    def __init__(self,numIn=128,numout=11):
        super(Lin,self).__init__()
        self.conv = nn.Conv2d(numIn,numout,1)
        self.bn = nn.BatchNorm2d(numout)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


class KFSGNet(nn.Module):

    def __init__(self):
        super(KFSGNet,self).__init__()
        self.__hg = Hourglass()
        self.__lin = Lin()
    def forward(self,x):
        x = self.__hg(x)
        x = self.__hg(x)
        x = self.__lin(x)
        return x

