import torch
import torch.nn as nn
import math


class Net(torch.nn.Module):
    def __init__(self, num_channels, base_filter, upscale_factor):
        super(Net, self).__init__()

        #这一部分的话可以认为是Coarse SR Network
        
        self.Coarse_SR_input_conv = nn.Conv2d(num_channels, base_filter, kernel_size=3, stride=1, padding=1)


        resnet_blocks = []
        for _ in range(4):
            resnet_blocks.append(ResnetBlock(base_filter, kernel=3, stride=1, padding=1))
        self.Coarse_SR_residual_layers = nn.Sequential(*resnet_blocks)

        self.Coarse_SR_mid_conv = nn.Conv2d(base_filter, base_filter, kernel_size=3, stride=1, padding=1)

        upscale = []
        for _ in range(int(math.log2(upscale_factor))):
            upscale.append(PixelShuffleBlock(base_filter, base_filter, upscale_factor=2))
        self.Coarse_SR_upscale_layers = nn.Sequential(*upscale)

        self.Coarse_SR_output_conv = nn.Conv2d(base_filter, num_channels, kernel_size=3, stride=1, padding=1)


        #这一部分的话做的是Fine SR Encoder
        #这里的话理论来说我们是要做Feature Map的映射然后再进行缩减的，但是这里的话没有做这一步，会不会就是因为这个导致我们后期的运算压力太大
        self.Fine_SR_input_conv = nn.Conv2d(num_channels, base_filter, kernel_size=3, stride=1, padding=1)
        
        resnet_blocks = []
        for _ in range(12):
            resnet_blocks.append(ResnetBlock(base_filter, kernel=3, stride=1, padding=1))
        self.Fine_SR_residual_layers = nn.Sequential(*resnet_blocks)


        self.Fine_SR_out_conv = nn.Conv2d(base_filter, num_channels, kernel_size=3, stride=1, padding=0)

        #在后面的话我们来做Fine SR Decoder
        self.Fine_SR_Decoder_input_conv = nn.Conv2d(base_filter, base_filter, kernel_size=3, stride=1, padding=0)
        
        self.Fine_SR_Decoder_deconv = nn.ConvTranspose2d(base_filter,base_filter *2,kernel_size = 4,stride = 1)
        

        resnet_blocks = []
        for _ in range(3):
            resnet_blocks.append(ResnetBlock(base_filter*2, kernel=3, stride=1, padding=1))
        self.Fine_SR_Decoder_residual_layers = nn.Sequential(*resnet_blocks)


        self.Fine_SR_Decoder_out_conv = nn.Conv2d(base_filter*2, num_channels, kernel_size=3, stride=1, padding=1)





    #这个的话是建立我们的前向传播模型
    def forward(self, x):
        #coarse SR
        x = self.Coarse_SR_input_conv(x)

        residual = x
        x = self.Coarse_SR_residual_layers(x)
        x = torch.add(x,residual)
    
        x =self.Coarse_SR_mid_conv(x)
        x = self.Coarse_SR_upscale_layers(x)
        x = self.Coarse_SR_output_conv(x) 

        # Fine SR Encoder
        x = self.Fine_SR_input_conv(x)
        x = self.Fine_SR_residual_layers(x)
        x = self.Fine_SR_out_conv(x)


        # Fine SR Decoder
        x = self.Fine_SR_Decoder_input_conv(x)
        x = self.Fine_SR_Decoder_deconv(x)
        x = self.Fine_SR_Decoder_residual_layers(x)
        x = self.Fine_SR_Decoder_out_conv(x)

        return x

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

#这个模块办到的是建立对应的resbolck
class ResnetBlock(nn.Module):
    def __init__(self, num_channel, kernel=3, stride=1, padding=1):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, num_channel, kernel, stride, padding)
        self.conv2 = nn.Conv2d(num_channel, num_channel, kernel, stride, padding)
        #self.bn = nn.BatchNorm2d(num_channel)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        #x = self.bn(self.conv1(x)）
        x = self.conv1(x)
        #下一步需要看这个
        x = self.activation(x)
        #x = self.bn(self.conv2(x))
        x = self.conv2(x)
        x = torch.add(x, residual)
        return x

class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upscale_factor, kernel=3, stride=1, padding=1):
        super(PixelShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel * upscale_factor ** 2, kernel, stride, padding)
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.ps(self.conv(x))
        return x
