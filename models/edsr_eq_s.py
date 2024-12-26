# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
# import torch.nn.functional as F
from models import B_Conv as fn

from models import register


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class w_diff(nn.Conv2d):
    def __init__(self, ColorW=(0.299, 0.587, 0.114)):

        super(w_diff, self).__init__(3, 3, kernel_size=3)
        TVkernel = (torch.Tensor([-1,2,-1]).view(1,1,1,3))*(torch.Tensor((0,1,0)).view(1,1,3,1))
        self.weight.data = (torch.Tensor(ColorW).view(1, 3, 1, 1))*TVkernel
        self.bias.data = torch.zeros(1)
        self.padding = 1
        for p in self.parameters():
            p.requires_grad = False

class h_diff(nn.Conv2d):
    def __init__(self, ColorW=(0.299, 0.587, 0.114)):

        super(h_diff, self).__init__(3, 3, kernel_size=1)
        TVkernel = (torch.Tensor([-1,2,-1]).view(1,1,3,1))*(torch.Tensor((0,1,0)).view(1,1,1,3))
        self.weight.data = (torch.Tensor(ColorW).view(1, 3, 1, 1))*TVkernel
        self.bias.data = torch.zeros(1)
        self.padding = 1
        for p in self.parameters():
            p.requires_grad = False
            
class eq_diff(nn.Conv2d):
    def __init__(self, ColorW=[0.299, 0.587, 04.11]):

        super(eq_diff, self).__init__(3,1,kernel_size=3)
        TVkernel = (torch.Tensor([[-0.7071,-1, -0.7071],[-1,6.827,-1],[-0.7071,-1, -0.7071]]).view(1,1,3,3))        
        self.weight.data = (torch.Tensor(ColorW).view(1, 3, 1, 1))*TVkernel
        self.bias.data = torch.zeros(1)
        self.padding = 1
        for p in self.parameters():
            p.requires_grad = False 
            
class R2E(nn.Module):
    def __init__(self, tranNum):
        super(R2E, self).__init__()
        self.tranNum = tranNum
    def forward(self, x):
        x = x.unsqueeze(2)
        # print(x.shape)
        x = x.repeat([1,1,self.tranNum,1,1])
        # print(x.shape)
        x = x.reshape(x.shape[0],-1, x.shape[-2], x.shape[-1])
        return x

# import scipy.io as sio    
class ScaleEncode(nn.Module):
    def __init__(self):

        super(ScaleEncode, self).__init__()
        self.eq_diff = eq_diff()
    def forward(self, x, scale):
        diffx = torch.cat((x, self.eq_diff(x)/scale/10), dim = 1)
        # sio.savemat('save/rotDiffObserve.mat',{'X': diffx.cpu().numpy(), 'x':x.cpu().numpy()})
        return diffx
    



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
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = args.kernel_size
        tranNum = args.tranNum
        scale = args.scale[0]
        act = nn.ReLU(True)
        iniScale = 0.1
        inP = kernel_size
        Smooth = False
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        # self.sub_mean = MeanShift(args.rgb_range)
        # self.add_mean = MeanShift(args.rgb_range, sign=1)

        in_dim = args.n_colors

        # define head module

        if args.scale_encode:
            self.ScaleEncode = ScaleEncode()
            m_head = [fn.Fconv_PCA(kernel_size,in_dim+1,n_feats//tranNum,tranNum,inP=inP,padding=(kernel_size-1)//2, ifIni=1, Smooth = Smooth, iniScale = iniScale)]
        else:
            m_head = [fn.Fconv_PCA(kernel_size,in_dim,n_feats//tranNum,tranNum,inP=inP,padding=(kernel_size-1)//2, ifIni=1, Smooth = Smooth, iniScale = iniScale)]

        # define body module
        m_body = [
            fn.ResBlock(
                fn.Fconv_PCA, n_feats//tranNum, kernel_size,tranNum = tranNum, inP = inP,  act=act, res_scale=args.res_scale, Smooth = Smooth, iniScale = iniScale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(fn.Fconv_PCA(kernel_size,n_feats//tranNum,n_feats//tranNum,tranNum,inP=inP,padding=(kernel_size-1)//2, ifIni=0, Smooth = Smooth, iniScale = iniScale))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            
            if args.thetaTail:
                self.invFeat = fn.Fconv_PCA_out(kernel_size,n_feats//tranNum,n_feats,tranNum,inP=inP,padding=(kernel_size-1)//2, ifIni=0, Smooth = Smooth, iniScale = iniScale)
                self.equFeat = fn.Fconv_PCA(kernel_size,n_feats//tranNum,1,tranNum,inP=inP,padding=(kernel_size-1)//2, ifIni=0, Smooth = Smooth, iniScale = iniScale)
                self.out_dim = n_feats+tranNum
            else:
                self.out_dim = n_feats
        
        else:
            self.out_dim = args.n_colors
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)
        

    def forward(self, x, scale = None):

        if self.args.scale_encode:
            scale=torch.mean(scale,dim  = 1).view([-1,1,1,1])
            x = self.ScaleEncode(x, scale)

        x = self.head(x)

        res = self.body(x)
        res += x

        if self.args.no_upsampling:
            if self.args.thetaTail:
                x = torch.cat([self.equFeat(res),self.invFeat(res)], dim = 1)
            else:
                x = res
        else:
            x = self.tail(res)
            
            
        #x = self.add_mean(x)
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
                    


@register('edsr-eq-s-baseline')
def make_edsr_baseline(n_colors = 3, scale_encode = False, n_resblocks=16, n_feats=64, res_scale=1,
                       scale=2, no_upsampling=False, rgb_range=1, tranNum = 4, kernel_size = 3,
                       thetaTail = False):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    
    args.kernel_size = kernel_size
    args.tranNum = tranNum

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    # args.max_scale = max_scale

    args.rgb_range = rgb_range
    args.n_colors = n_colors
    args.scale_encode = scale_encode
    args.thetaTail = thetaTail
    return EDSR(args)


# @register('edsr-fconv')
# def make_edsr(n_resblocks=32, n_feats=256, res_scale=0.1,
#               scale=2, no_upsampling=False, rgb_range=1):
#     args = Namespace()
#     args.n_resblocks = n_resblocks
#     args.n_feats = n_feats
#     args.res_scale = res_scale

#     args.scale = [scale]
#     args.no_upsampling = no_upsampling

#     args.rgb_range = rgb_range
#     args.n_colors = 3
#     return EDSR(args)
