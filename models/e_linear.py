# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:58:33 2021

@author: XieQi
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import models

class EQ_linear_input(nn.Module):
    def __init__(self, inNum, outNum, tranNum=8, bias=True, iniScale = 1.0, corrd_scale = 1.0):
       
        super(EQ_linear_input, self).__init__()
        self.linear = EQ_linear_inter(inNum+2, outNum, tranNum, bias, iniScale)
        self.corrd_scale = corrd_scale
        self.tranNum = tranNum
        self.inNum = inNum
        self.GetTranMatrx()

    def forward(self, input):
        coord = input[:,-2:]*self.corrd_scale
        x = coord[:,0].unsqueeze(1).unsqueeze(2)
        y = coord[:,1].unsqueeze(1).unsqueeze(2)
        feat  = input[:,:-2].reshape([-1, self.inNum, self.tranNum]) #BxnxtranNum
        TC_x = self.cosTheta*x-self.sinTheta*y #Bx1xtranNum
        TC_y = self.cosTheta*y+self.sinTheta*x #Bx1xtranNum
        inputF  = torch.cat([feat, TC_x, TC_y], dim = 1)#Bx(n+2)*tranNum
        inputF  = inputF.reshape(-1, (self.inNum+2)*self.tranNum)
        output = self.linear(inputF)
        return output

    def GetTranMatrx(self):
        theta = torch.arange(self.tranNum)/self.tranNum*2*np.pi
        theta = -theta.reshape([1,1,self.tranNum]) # -theta is due to the different of np.meshgrid and torch.meshgrid
        # self.cosTheta = torch.cos(theta)
        # self.sinTheta = torch.sin(theta)
        self.register_buffer("cosTheta", torch.cos(theta))
        self.register_buffer("sinTheta", torch.sin(theta))

class EQ_linear_inter(nn.Module):
    
    def __init__(self, inNum, outNum, tranNum=8, bias=True, iniScale = 1.0):
       
        super(EQ_linear_inter, self).__init__()

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.weights = nn.Parameter(torch.Tensor(outNum, 1, inNum, tranNum), requires_grad=True)
        
        # iniw = Getini_reg(inNum, outNum, tranNum)*iniScale #(outNum,1,inNum,expand)
        # self.weights = nn.Parameter(iniw, requires_grad=True)

        self.padding = 0
        self.bias = bias

        if bias:
            self.c = nn.Parameter(torch.Tensor(outNum,1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()


    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        tempW = self.weights.repeat([1,tranNum,1,1])

        tempWList = [torch.cat([tempW[:,i:i+1,:,-i:],tempW[:,i:i+1,:,:-i]], dim = 3) for i in range(tranNum)]   
        tempW = torch.cat(tempWList, dim = 1)

        weight = tempW.reshape([outNum*tranNum, inNum*tranNum])
        if self.bias:
            bias = self.c.repeat([1,tranNum]).reshape([1,outNum*tranNum])#.cuda()
        else:
            bias = self.c

        return F.linear(input, weight, bias = bias)
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)

class EQ_linear_output(nn.Module):
    def __init__(self, inNum, outNum, tranNum=8, bias=True, iniScale = 1.0):
       
        super(EQ_linear_output, self).__init__()

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum,1), requires_grad=True)
        
        self.padding = 0
        self.bias = bias

        if bias:
            self.c = nn.Parameter(torch.Tensor(outNum))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()

    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        tempW = self.weights.repeat([1,1,tranNum])

        weight = tempW.reshape([outNum, inNum*tranNum])

        return F.linear(input, weight, bias = self.c)
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)

class Dropout(nn.Module):
    #这个之后可以放到e_linear里面去
    def __init__(self, p = 0.,  tranNum=4):
        # nn.Dropout2d
        super(Dropout, self).__init__()
        self.tranNum = tranNum
        self.Dropout = nn.Dropout2d(p)
    def forward(self, X):
        sizeX = X.shape
        X = self.Dropout(X.reshape([-1, sizeX[-1]//self.tranNum, self.tranNum]))
        return X.reshape(sizeX)

def Getini_reg(inNum, outNum, expand): 
    A = (np.random.rand(outNum,1,inNum,expand)-0.5)*2*2.4495/np.sqrt((inNum))
    return torch.FloatTensor(A)


class EQ_lte_input(nn.Module):
    def __init__(self, tranNum=8, corrd_scale = 1):
       
        super(EQ_lte_input, self).__init__()
        self.tranNum = tranNum
        self.GetTranMatrx()
        self.corrd_scale = corrd_scale

    def forward(self, freq, coef, phase, coord):
        tranNum = self.tranNum
        bs,q = freq.shape[:2]
        coord = coord*self.corrd_scale
        x = coord[:,0].unsqueeze(1).unsqueeze(2)
        y = coord[:,1].unsqueeze(1).unsqueeze(2)
        rot_x = self.cosTheta*x-self.sinTheta*y #Bx1xtranNum
        rot_y = self.cosTheta*y+self.sinTheta*x #Bx1xtranNum
        rot_corrd = torch.cat([rot_x, rot_y], dim = 1) #Bx2xtranNum
        # print(q/2/tranNum)
        # print(q)
        freq = freq.reshape([bs, 2, q//2//tranNum, tranNum ]) #Bx2xkxtranNum, k=q/2/tranNums

        # coef = coef.reshape([bs, q/2, 2, tranNum ])
        outp = torch.einsum('bckt,bct->bkt', freq, rot_corrd) #BxkxtranNums
        outp = outp.reshape([bs, q//2]) 
        outp += phase
        outp = torch.cat((torch.cos(np.pi*outp), torch.sin(np.pi*outp)), dim=-1) #Bxq
        outp = outp*coef #Bxq
        # outp = coef

        return outp

    def GetTranMatrx(self):
        theta = torch.arange(self.tranNum)/self.tranNum*2*np.pi
        theta = -theta.reshape([1,1,self.tranNum]) # -theta is due to the different of np.meshgrid and torch.meshgrid
        # self.cosTheta = torch.cos(theta)
        # self.sinTheta = torch.sin(theta)
        self.register_buffer("cosTheta", torch.cos(theta))
        self.register_buffer("sinTheta", torch.sin(theta))
        
        
class EQ_OPE_input(nn.Module):
    def __init__(self, tranNum=8, imnet_spec=None, corrd_scale = 1):
       
        super(EQ_OPE_input, self).__init__()
        self.tranNum = tranNum
        self.GetTranMatrx()
        self.corrd_scale = corrd_scale
        if imnet_spec is not None:
            self.imnet = models.make(imnet_spec)
        else:
            self.imnet = None

    def forward(self, x):
        x = x.view([-1, x.shape[-1]])
        coord = x[:,-2:].reshape(-1,1,2)*self.corrd_scale
        feat  = x[:,:-2].reshape([-1,(x.shape[-1]-2)//self.tranNum, self.tranNum]) #BxnxtranNum
        feat  = feat.permute([0,2,1])#BxtranNumxn
        TC_x = self.cosTheta*coord[:,:,0]-self.sinTheta*coord[:,:,1] #BxtranNum
        TC_y = self.cosTheta*coord[:,:,1]+self.sinTheta*coord[:,:,0] #BxtranNum
        TC_x = TC_x.unsqueeze(2)#BxtranNumx1
        TC_y = TC_y.unsqueeze(2)#BxtranNumx1
        inputF  = torch.cat([feat, TC_x, TC_y], dim = 2)#BxtranNumx(n+2)
        inputF  = inputF.reshape([-1,inputF.shape[-1]])#BtranNumx(n+2)
        outputF = self.imnet(inputF)
        outputF = outputF.reshape([-1, self.tranNum, outputF.shape[-1]])
        return torch.sum(outputF,dim = 1)/self.tranNum

    def GetTranMatrx(self):
        theta = torch.arange(self.tranNum)/self.tranNum*2*np.pi
        theta = -theta.reshape([1,self.tranNum]) # -theta is due to the different of np.meshgrid and torch.meshgrid
        # self.cosTheta = torch.cos(theta)
        # self.sinTheta = torch.sin(theta)
        self.register_buffer("cosTheta", torch.cos(theta))
        self.register_buffer("sinTheta", torch.sin(theta))

