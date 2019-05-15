import torch
import torch.nn as nn
import math
import pickle
from torch.autograd import Variable
from models import KFSGNet

def loss_MSE(x, y, size_average=False):
  z = x - y 
  z2 = z * z
  if size_average:
    return z2.mean()
  else:
    return z2.sum().div(x.size(0)*2)
    
def loss_Textures(x, y, nc=3, alpha=1.2, margin=0):
  xi = x.contiguous().view(x.size(0), -1, nc, x.size(2), x.size(3))
  yi = y.contiguous().view(y.size(0), -1, nc, y.size(2), y.size(3))
  
  xi2 = torch.sum(xi * xi, dim=2)
  yi2 = torch.sum(yi * yi, dim=2)
  
  out = nn.functional.relu(yi2.mul(alpha) - xi2 + margin)
  
  return torch.mean(out)

class _Residual_Block(nn.Module):

    def __init__(self, inc=64, outc=64, groups=1):
        super(_Residual_Block, self).__init__()
        
        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
          self.conv_expand = None
          
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x): 
        if self.conv_expand is not None:
          identity_data = self.conv_expand(x)
        else:
          identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output,identity_data)))
        return output 

def make_layer(block, num_of_layer, inc=64, outc=64, groups=1):
    layers = []
    layers.append(block(inc=inc, outc=outc, groups=groups))
    for _ in range(1, num_of_layer):
        layers.append(block(inc=outc, outc=outc, groups=groups))
    return nn.Sequential(*layers)   

class _Interim_Block(nn.Module): 
    def __init__(self, inc=64, outc=64, groups=1):
        super(_Interim_Block, self).__init__()
        
        self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.ReLU(inplace=True)
     
    def forward(self, x): 
        identity_data = self.conv_expand(x)          
        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output,identity_data)))
        return output    
        
class NetSR(nn.Module):
    def __init__(self, scale=2, num_layers_res=2):
        super(NetSR, self).__init__()
        

        #----------input conv-------------------
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(64) 
        self.relu_input = nn.ReLU(inplace=True)
        
        #----------residual-------------------
        self.residual = nn.Sequential(
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64)
        )
        self.conv_input_out1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_input_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_input_1 = nn.BatchNorm2d(64)
        self.relu_input_1 = nn.ReLU(inplace=True)

        self.residual1 = nn.Sequential(
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64)
        )

        self.conv_input_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_input_2 = nn.BatchNorm2d(64)
        self.relu_input_2 = nn.ReLU(inplace=True)

        self.conv_input_3 = nn.Conv2d(in_channels=75, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_input_3 = nn.BatchNorm2d(64)
        self.relu_input_3 = nn.ReLU(inplace=True)

        self.conv_input_4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn_input_4 = nn.BatchNorm2d(64)
        self.relu_input_4 = nn.ReLU(inplace=True)

        self.residual2 = nn.Sequential(
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64)
        )

        self.conv_input_out2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_input_hg = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_input_hg = nn.BatchNorm2d(64)
        self.relu_input_hg = nn.ReLU(inplace=True)

        self.residual3 = nn.Sequential(
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=128),
            make_layer(_Residual_Block, num_layers_res, inc=128, outc=128),
            make_layer(_Residual_Block, num_layers_res, inc=128, outc=128)
        )

        self.HG = KFSGNet()
        
    def forward(self, x):
        
        f = self.relu_input(self.bn_input(self.conv_input(x)))
        f = self.residual(f)
        out1 = self.conv_input_out1(f)
        f1 = self.relu_input_1(self.bn_input_1(self.conv_input_1(out1)))
        f1 = self.residual1(f1)
        f1 = self.relu_input_2(self.bn_input_2(self.conv_input_2(f1)))
        f2 = self.relu_input_hg(self.bn_input_hg(self.conv_input_hg(out1)))
        f2 = self.residual3(f2)
        out2 = self.HG(f2)
        f3 = torch.cat((f1,out2),1)
        f3 = self.relu_input_3(self.bn_input_3(self.conv_input_3(f3)))
        f3 = self.relu_input_4(self.bn_input_4(self.conv_input_4(f3)))
        f3 = self.residual2(f3)
        out3 = self.conv_input_out2(f3)

        return out1,out2,out3
         