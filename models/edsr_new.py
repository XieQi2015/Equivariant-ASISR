# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=3)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class w_diff(nn.Conv2d):
    def __init__(self, ColorW=(0.299, 0.587, 0.114)):

        super(w_diff, self).__init__(3, 3, kernel_size=3)
        TVkernel = (torch.Tensor((-1,2,-1)).view(1,1,1,3))*(torch.Tensor((0,1,0)).view(1,1,3,1))
        self.weight.data = (torch.Tensor(ColorW).view(1, 3, 1, 1))*TVkernel
        self.bias.data = torch.zeros(1)
        self.padding = 1
        for p in self.parameters():
            p.requires_grad = False

class h_diff(nn.Conv2d):
    def __init__(self, ColorW=(0.299, 0.587, 0.114)):

        super(h_diff, self).__init__(3, 3, kernel_size=1)
        TVkernel = (torch.Tensor((-1,2,-1)).view(1,1,3,1))*(torch.Tensor((0,1,0)).view(1,1,1,3))
        self.weight.data = (torch.Tensor(ColorW).view(1, 3, 1, 1))*TVkernel
        self.bias.data = torch.zeros(1)
        self.padding = 1
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    
    
class NoResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
#        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        print('EDSR Cell Decoder Version')
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        outputdim = args.outputdim
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.h_diff = h_diff()
        self.w_diff = w_diff()
        if args.cell_decode:
            in_dim = args.n_colors+2
        else:
            in_dim = args.n_colors

        # define head module
        m_head = [conv(in_dim, n_feats, kernel_size),
                  ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)]


        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks-1)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        # self.m_res = nn.Sequential(*m_res)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = outputdim
            self.tail = conv(n_feats, outputdim, kernel_size)
            
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x, scale = None):
        #x = self.sub_mean(x)
        # sh = torch.abs(self.h_diff(x))+0.9
        # sw = torch.abs(self.w_diff(x))+0.9
        if self.args.cell_decode:
            #scale_h=torch.min(scale[:,0].view([-1,1,1,1]),torch.ones(1).cuda()*4.0)
            #scale_w=torch.min(scale[:,1].view([-1,1,1,1]),torch.ones(1).cuda()*4.0)
            scale = 1+torch.exp((2-scale)*2)
            scale_h=scale[:,0].view([-1,1,1,1])
            scale_w=scale[:,1].view([-1,1,1,1])
            x = torch.cat([x, self.h_diff(x)*scale_h, self.w_diff(x)*scale_w], dim = 1)
        x = self.head(x)
        
        res = self.body(x)
        res += x

        x = self.tail(res)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
                    


@register('edsr-NewCD-baseline')
def make_edsr_cd_baseline(n_colors = 3, cell_decode = False, n_resblocks=16, n_feats=64, outputdim = 64,  res_scale=1,
                       scale=2, no_upsampling=False, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.outputdim = outputdim
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = n_colors
    args.cell_decode = cell_decode
    return EDSR(args)


