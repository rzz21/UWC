import torch
import torch.nn as nn
from collections import OrderedDict

from utils import logger

__all__ = ["SPPNet"]

r''' model.py
用什么模型待确定
'''

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, pad=True, stride=1, groups=1):
        if pad:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class SPPBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, res=False):
        super(SPPBlock, self).__init__()
        self.channel = mid_channel
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(in_channel, self.channel, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(self.channel, self.channel, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(self.channel, self.channel, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x3', ConvBN(in_channel, self.channel, [1, 3])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x1', ConvBN(self.channel, self.channel, [3, 1])),
        ]))
        self.conv1x1_1 = ConvBN(self.channel * 2, in_channel, 1)
        self.conv1x1_2 = ConvBN(in_channel, out_channel, 1)
        if res:
            self.identity = nn.Identity()
        else:
            self.identity = None
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x) if self.identity else None

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1_1(out)

        if identity is not None:
            out = self.relu(out + identity)
        else:
            out = self.relu(out)

        out = self.relu(self.conv1x1_2(out))
        return out
      
class ContractBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, pad=True, stride=1, groups=1, pool_kernel=2, pool_stride=2):
        super(ContractBlock, self).__init__(OrderedDict([('max_pool', nn.MaxPool2d(pool_kernel, pool_stride)),
                                                         ('conv', ConvBN(in_channel, out_channel, kernel_size, pad=pad, stride=stride, groups=groups)),
                                                         ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True))]))

class ExpandBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, pad=True, stride=1, groups=1, upsample_kernel_size=2, upsample_stride=2):
        super(ExpandBlock, self).__init__(OrderedDict([('conv', ConvBN(in_channel, out_channel, kernel_size, pad=pad, stride=stride, groups=groups)),
                                                       ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                                                       ('deconv', nn.ConvTranspose2d(out_channel, out_channel, kernel_size=upsample_kernel_size, stride=upsample_stride))]))
        
# class BottleneckBlock(nn.Sequential):
#     def __init__(self, channel, sample_rate):
#         super(BottleneckBlock, self).__init__(OrderedDict([('max_pool', nn.MaxPool2d(sample_rate, sample_rate)),
                                                        #    ('deconv', nn.ConvTranspose2d(channel, channel, kernel_size=sample_rate, stride=sample_rate))]))


class SPPNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SPPNet, self).__init__()
        self.encoder_block = SPPBlock(in_channel, 8, 32) # 256*512*in_channel -> 256*512*32
        self.contract1 = ContractBlock(32, 64, pool_kernel=2, pool_stride=2) # 256*512*32 -> 128*256*64
        self.contract2 = ContractBlock(64, 128, pool_kernel=2, pool_stride=2) # 128*256*64 -> 64*128*128
        self.contract3 = ContractBlock(128, 256, pool_kernel=4, pool_stride=4) # 64*128*128 -> 16*32*256
        self.max_pool = nn.MaxPool2d(2, 2) # 16*32*256 -> 8*16*256

        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2) # 8*16*256 -> 16*32*256
        self.expand1 = ExpandBlock(512, 256, upsample_kernel_size=4, upsample_stride=4) # 16*32*256+16*32*256 -> 64*128*256
        self.expand2 = ExpandBlock(384, 128, upsample_kernel_size=2, upsample_stride=2) # 64*128*256+64*128*128 -> 128*256*128
        self.expand3 = ExpandBlock(192, 64, upsample_kernel_size=2, upsample_stride=2) # 128*256*128+128*256*64 -> 256*512*64
        self.conv1x1 = ConvBN(96, 2, 1) # 256*512*64 -> 256*512*2
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.decoder_block = SPPBlock(2, 4, out_channel, res=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encode
        encode_out1 = self.encoder_block(x)
        encode_out2 = self.contract1(encode_out1)
        encode_out3 = self.contract2(encode_out2)
        encode_out4 = self.contract3(encode_out3)

        # bottleneck
        bottleneck = self.deconv(self.max_pool(encode_out4))

        # decode
        decode_out1 = self.expand1(torch.cat((encode_out4, bottleneck), dim=1))
        decode_out2 = self.expand2(torch.cat((encode_out3, decode_out1), dim=1))
        decode_out3 = self.expand3(torch.cat((encode_out2, decode_out2), dim=1))
        
        # conv1x1
        out = self.conv1x1(torch.cat((encode_out1, decode_out3), dim=1))
        out = self.relu(out)
        out = self.decoder_block(out)
        out = self.sigmoid(out)

        return out
        pass