import torch
import torch.nn as nn

import models
from models import register


@register('eq_fun')
class EQ_fun(nn.Module):

    def __init__(self, parfun_spec=None, inP=None, tranNum = 4,in_dim = 256, coord_scale = 1.0):
        super().__init__()
        if parfun_spec is not None:
            self.imnet = models.make(parfun_spec, args={'in_dim':(in_dim-2)//tranNum+2, 'inP':inP})
        else:
            self.imnet = None
        self.tranNum = tranNum
        self.corrd_scale = coord_scale
        self.GetTranMatrx()

    def forward(self, x):
        x = x.view([-1, x.shape[-1]])
        coord = x[:,-2:].reshape(-1,1,2)*self.corrd_scale
        feat  = x[:,:-2].reshape([-1,(x.shape[-1]-2)//self.tranNum, self.tranNum]) #BxnxtranNum
        feat  = feat.permute([0,2,1])#BxtranNumxn
        TC_x = self.cosTheta*coord[:,:,0]-self.sinTheta*coord[:,:,1] #BxtranNum
        TC_y = self.cosTheta*coord[:,:,1]+self.sinTheta*coord[:,:,0] #BxtranNum
        TC_x = TC_x.unsqueeze(2)
        TC_y = TC_y.unsqueeze(2)
        inputF  = torch.cat([feat, TC_x, TC_y], dim = 2)#BxtranNumx(n+2)
        inputF  = inputF.reshape([-1,inputF.shape[-1]])#BtranNumx(n+2)
        outputF = self.imnet(inputF)
        outputF = outputF.reshape([-1, self.tranNum, outputF.shape[-1]])
        return torch.sum(outputF,dim = 1)/self.tranNum
    
    def GetTranMatrx(self):
        theta = torch.arange(self.tranNum)/self.tranNum*2*torch.pi
        theta = -theta.reshape([1,self.tranNum])
        # self.cosTheta = torch.cos(theta)
        # self.sinTheta = torch.sin(theta)
        self.register_buffer("cosTheta", torch.cos(theta))
        self.register_buffer("sinTheta", torch.sin(theta))

