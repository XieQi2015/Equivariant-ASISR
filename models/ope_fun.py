import torch.nn as nn
import torch
import numpy as np
import math

from models import register

@register('Feiuier')
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
    
    
    
   

def get_embed_fns(max_freq):
    """
    N,bsize,1 ---> N,bsize,2n+1
    """
    embed_fns = []
    embed_fns.append(lambda x: torch.ones(x.shape))  # x: N,bsize,1
    for i in range(1, max_freq + 1):
        embed_fns.append(lambda x, freq=i: math.sqrt(2) * torch.cos(x * freq))
        embed_fns.append(lambda x, freq=i: math.sqrt(2) * torch.sin(x * freq))
    return embed_fns


class OPE(nn.Module):
    def __init__(self, max_freq, omega):
        super(OPE, self).__init__()
        self.max_freq = max_freq
        self.omega = omega
        self.embed_fns = get_embed_fns(self.max_freq)

    def embed(self, inputs):
        """
        N,bsize,1 ---> N,bsize,1,2n+1
        """
#        print(inputs.shape)
#        print(self.embed_fns[0](inputs * self.omega).shape)
        res = torch.cat([fn(inputs * self.omega).to(inputs.device) for fn in self.embed_fns], -1)
        return res.unsqueeze(-2)

    def forward(self, coords):
        """
        N,bsize,2 ---> N,bsize,(2n+1)^2
        """
        x_coord = coords[:, 0].unsqueeze(-1) #B*1*1
        y_coord = coords[:, 1].unsqueeze(-1) #B*1*1
        X = self.embed(x_coord) #B*1*N
        Y = self.embed(y_coord) #B*1*N
        ope_mat = torch.matmul(X.transpose(-2, -1), Y) #B*N*N
        ope_flat = ope_mat.view(ope_mat.shape[0], -1) #B*NN 
        return ope_flat
    


@register('ope_fun')
class LC_OPE(nn.Module):
    """
    linear combination of OPE with 3 channels
    """

    def __init__(self, max_freq=3, omega=0.5 * math.pi):
        super(LC_OPE, self).__init__()
        self.max_freq = max_freq
        self.omega = omega
        self.c = (2 * max_freq + 1) ** 2
        self.ope = OPE(max_freq=max_freq, omega=omega)

    def forward(self, x):
        """
        N,bsize,ccc N,bsize,ccc ---> N,bsize,3
        """

        x = x.view([-1, x.shape[-1]])
        rel_coord = x[:,-2:] #
        latent = x[:,:-2]
        
        c = int(latent.shape[-1] // 3)
        assert c == self.c
        
        ope_flat = self.ope(rel_coord).unsqueeze(-2)
        latent_R = latent[:,  :c].unsqueeze(-1)
        latent_G = latent[:,  c:c * 2].unsqueeze(-1)
        latent_B = latent[:,  c * 2:].unsqueeze(-1)
        R = torch.matmul(ope_flat, latent_R).squeeze(-1)
        G = torch.matmul(ope_flat, latent_G).squeeze(-1)
        B = torch.matmul(ope_flat, latent_B).squeeze(-1)
        ans = torch.cat([R, G, B], dim=-1)
        return ans/100 
