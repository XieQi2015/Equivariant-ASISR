import torch.nn as nn
import torch
import numpy as np

from models import register

@register('f_fun_rot3')
class F_Fun(nn.Module):

    def __init__(self, out_dim, inP = 5):
        super().__init__()
        self.inP = inP
        k1, k2, l1, l2 = self.GetKL()
        self.k1  = k1.cuda()
        self.k2  = k2.cuda()
        self.l1  = l1.cuda()
        self.l2  = l2.cuda()
        self.out_dim = out_dim
        return 

    def GetKL(self):
        inP = self.inP

        k1  = -torch.arange(0,inP//2)
        k2  = -torch.arange(1,inP//2)
        l1  = -torch.arange(0,inP//2)
        l2  =  torch.arange(1,inP//2)
        
        # k1  = -torch.arange(0,inP//2)
        # k2  = -torch.arange(1,inP//2)
        # l1  = -torch.arange(0,inP//2)
        # l2  =  torch.arange(1,inP//2)
        
        k1  = torch.cat([k1, -torch.ones(1)/2], dim = 0)
        k2  = torch.cat([k2, -torch.ones(1)/2], dim = 0)
        l1  = torch.cat([l1, -torch.ones(1)/2], dim = 0)
        l2  = torch.cat([l2,  torch.ones(1)/2], dim = 0)

        
        inP = self.inP
        v  = np.pi/inP*(inP-1)
        k1  = k1.reshape([1, inP//2+1,1])*v
        l1  = l1.reshape([1, 1,inP//2+1])*v
        k2  = k2.reshape([1, inP//2,1])*v
        l2  = l2.reshape([1, 1,inP//2])*v
        return k1, k2, l1, l2
    
    def iFun(self, feat, x):
        B,N = feat.shape
        inP = self.inP
        
        fpar = feat[:,-5:] # 
#        print(feat.shape)
        feat = feat[:,:-5].reshape([B, inP*inP, self.out_dim])
        X = x[:, 0]+fpar[:,3]
        Y = x[:, 1]+fpar[:,4]
        
        X = torch.cos(fpar[:,0])*X-torch.sin(fpar[:,0])*Y
        Y = torch.sin(fpar[:,0])*X+torch.cos(fpar[:,0])*Y
        X = (fpar[:,1]+1)*X
        Y = (fpar[:,2]+1)*Y
        
        X = X.view([B,1,1])
        Y = Y.view([B,1,1])
        
        Basis =  torch.cat([torch.cos(self.k1*X+self.l1*Y).reshape(-1,(inP//2+1)*(inP//2+1)),
                            torch.cos(self.k2*X+self.l2*Y).reshape(-1,(inP//2)*(inP//2)), 
                            torch.sin(self.k1*X+self.l2*Y).reshape(-1,(inP//2)*(inP//2+1)),
                            torch.sin(self.k2*X+self.l1*Y).reshape(-1,(inP//2)*(inP//2+1))], dim = 1) #BxPP
        
        return torch.einsum('bin,bi->bn', feat, Basis)

    def forward(self, x):
        shape = x.shape[:-1]
        x = x.view([-1, x.shape[-1]])
        X = x[:,-2:]/2 #
        feat = x[:,:-2]
#        feat = self.layers(feat)
        oup = self.iFun(feat, X)
        
        return oup.view(*shape, -1)
    
    
    
    
