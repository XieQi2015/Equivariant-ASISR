import torch.nn as nn
from models import e_linear as en
from models import register


@register('e_mlp_lte')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, inP = None, tranNum = None):
        super().__init__()
        layers = []
        lastv = in_dim//tranNum
        layers.append(en.EQ_linear_output(lastv, hidden_list[0], tranNum))
        lastv = hidden_list[0]
        for hidden in hidden_list[1:]:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)
        

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)
