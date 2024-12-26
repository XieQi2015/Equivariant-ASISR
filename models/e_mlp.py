import torch.nn as nn
from models import e_linear as en
from models import register


@register('e_mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, inP = None, tranNum = None):
        super().__init__()
        layers = []
        lastv = (in_dim-2)//tranNum
        layers.append(en.EQ_linear_input(lastv, hidden_list[0]//tranNum, tranNum, corrd_scale = 0.1))
        layers.append(nn.ReLU())
        lastv = hidden_list[0]//tranNum
        for hidden in hidden_list[1:]:
            layers.append(en.EQ_linear_inter(lastv, hidden//tranNum, tranNum))
            layers.append(nn.ReLU())
            lastv = hidden//tranNum
        layers.append(en.EQ_linear_output(lastv, out_dim, tranNum))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)
