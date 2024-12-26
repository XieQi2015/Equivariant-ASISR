import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from models import register


@register('eq_fun_theta_2')
class EQ_fun(nn.Module):

    def __init__(self, parfun_spec=None, inP=None, tranNum = 4,in_dim = 256):
        super().__init__()
        if parfun_spec is not None:
            self.imnet = models.make(parfun_spec, args={'in_dim':(in_dim-tranNum), 'inP':inP})
        else:
            self.imnet = None
        self.tranNum = tranNum
        self.GetThetaBook()


    def forward(self, x):
        x = x.view([-1, x.shape[-1]])
        coord  = x[:,-2:]
        rotW = x[:,:self.tranNum]
        feat   = x[:,self.tranNum:-2]
        rotW = F.softmax(rotW,dim=1)
        # rotW2 = rotW
        # rotW2[:,0] += 0.01 
        # rotInd = torch.argmax(rotW2, dim = 1).unsqueeze(1)
        # n_times = torch.div(self.indBook-rotInd+self.tranNum/2, self.tranNum, rounding_mode  = 'floor')
        # n_res   = self.indBook-rotInd+self.tranNum/2 - n_times*self.tranNum
        # rotInd = n_res-self.tranNum/2+rotInd
        # theta = torch.sum(rotInd*rotW, dim=1)*self.delta
        # rotW[:,0] += 0.00256
        rotInd = torch.argmax(rotW, dim = 1)
        # print(rotInd)
        theta = rotInd*self.delta
        TC_y = torch.cos(theta)*coord[:,0]+torch.sin(theta)*coord[:,1] 
        TC_x = torch.cos(theta)*coord[:,1]-torch.sin(theta)*coord[:,0] 
        feat = torch.cat([feat, TC_y.unsqueeze(1),TC_x.unsqueeze(1)], dim=1)
        # feat   = x[:,self.tranNum:]
        outputF = self.imnet(feat)
        return outputF
    
    def GetThetaBook(self):
         indBook = torch.arange(self.tranNum).reshape([1,self.tranNum])
         self.register_buffer("indBook", indBook)
         # self.register_buffer("delta", torch.FloatTensor(2/self.tranNum)*torch.pi)
         self.delta = 2*torch.pi/self.tranNum
    #     # self.cosTheta = torch.cos(theta)
    #     # self.sinTheta = torch.sin(theta)
    #     self.register_buffer("cosTheta", torch.cos(theta))
    #     self.register_buffer("sinTheta", torch.sin(theta))

