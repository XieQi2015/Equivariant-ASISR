import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
from models import B_Conv as fn
from models import e_linear as en



# import numpy as np

@register('lte_eq')
class LTE(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256, local_ensemble=True, tranNum = 1, kernel_size = 5, upinput = True,  corrd_scale = 1):
        super().__init__()        
        # self.encoder = models.make(encoder_spec)
        self.local_ensemble = local_ensemble
        self.upinput = upinput
        self.corrd_scale = corrd_scale
        self.tranNum = tranNum
        self.encoder = models.make(encoder_spec, args = {'tranNum':tranNum})
        self.coef = fn.Fconv_PCA(kernel_size,self.encoder.out_dim//tranNum,hidden_dim//tranNum,tranNum=tranNum,padding=kernel_size//2)    
        self.freq = fn.Fconv_PCA(kernel_size,self.encoder.out_dim//tranNum,hidden_dim//tranNum,tranNum=tranNum,padding=kernel_size//2)
        self.phase   = en.EQ_linear_inter(2, hidden_dim//2//tranNum, tranNum, bias = False)#这里有
        self.freqLayer = en.EQ_lte_input(tranNum)
        self.imnet = models.make(imnet_spec, args={'tranNum': tranNum, 'in_dim':hidden_dim})

    def gen_feat(self, inp):
        self.inp = inp
        self.feat_coord = make_coord(inp.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:])
        
        feat = self.encoder(inp)
        self.coeff = self.coef(feat)
        self.freqq = self.freq(feat)

    def query_rgb(self, coord, cell=None):
        # feat = self.feat
        coef = self.coeff
        freq = self.freqq

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / coef.shape[-2] / 2
        ry = 2 / coef.shape[-1] / 2

        feat_coord = self.feat_coord

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_freq = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= coef.shape[-2]
                rel_coord[:, :, 1] *= coef.shape[-1]
                
                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= coef.shape[-2]
                rel_cell[:, :, 1] *= coef.shape[-1]
                rel_cellx = rel_cell[:, :, 0].unsqueeze(2).repeat([1,1,self.tranNum])
                rel_celly = rel_cell[:, :, 1].unsqueeze(2).repeat([1,1,self.tranNum])
                rel_cell_eq = torch.cat([rel_cellx, rel_celly], dim = -1)
                bs, q = coord.shape[:2]
                q_phase = self.phase(rel_cell_eq.reshape((bs * q, -1)))
                
                # basis generation
                inp = self.freqLayer(q_freq.reshape(bs*q, -1),q_coef.reshape(bs*q, -1),q_phase,rel_coord.reshape(bs*q, -1)*self.corrd_scale)
                # inp = torch.cat([q_coef.view(bs*q, -1), rel_coord.view(bs*q, -1)], dim=-1)
                # print(inp.shape)

                pred = self.imnet(inp).reshape(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        # t = areas[0]; areas[0] = areas[3]; areas[3] = t
        # t = areas[1]; areas[1] = areas[2]; areas[2] = t
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        if self.upinput:
            ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                          padding_mode='border', align_corners=False)[:, :, 0, :] \
                          .permute(0, 2, 1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
