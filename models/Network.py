import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as v

class MLP(nn.Module):
    def __init__(self,m,input,output):
        super(MLP, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(in_features=input,out_features=m),nn.Tanh(),
            nn.Linear(m,m),nn.Tanh(),
            nn.Linear(m,m),nn.Tanh(),
            nn.Linear(m,m),nn.Tanh(),
            nn.Linear(m,m),nn.Tanh(),
            nn.Linear(m,output)
        )
    def forward(self, x):
        return self.net(x)


class ResNet1(nn.Module):
    def __init__(self,m,input,out):
        super(ResNet1, self).__init__()
        self.input = nn.Linear(input,m)
        self.block1=nn.Sequential(
            nn.Linear(m,m),nn.ReLU(),
            nn.Linear(m,m),nn.ReLU(),
        )
        self.block2=nn.Sequential(
            nn.Linear(m,m),nn.ReLU(),
            nn.Linear(m,m),nn.ReLU(),
        )
        self.block3=nn.Sequential(
            nn.Linear(m,m),nn.ReLU(),
            nn.Linear(m,m),nn.ReLU(),
        )
        self.block4=nn.Sequential(
            nn.Linear(m,m),nn.ReLU(),
            nn.Linear(m,m),nn.ReLU(),
        )
        self.block5=nn.Sequential(
            nn.Linear(m,m),nn.ReLU(),
            nn.Linear(m,m),nn.ReLU(),
        )
        self.block6=nn.Sequential(
            nn.Linear(m,m),nn.ReLU(),
            nn.Linear(m,m),nn.ReLU(),
        )
        self.out = nn.Linear(m,out)
    def forward(self, x):
        x = self.input(x)
        x = self.block1(x) + x
        x = self.block2(x) + x
        x = self.block3(x) + x
        x = self.block4(x) + x
        x = self.block5(x) + x
        x = self.block6(x) + x
        x = self.out(x)
        return x


class ResNet2(nn.Module):
    def __init__(self,m,input,out):
        super(ResNet2, self).__init__()
        self.input = nn.Linear(input,m)
        self.block1=nn.Sequential(
            nn.Linear(m,m),nn.ReLU(),
            nn.Linear(m,m),nn.ReLU(),
        )
        self.block2=nn.Sequential(
            nn.Linear(m,m),nn.ReLU(),
            nn.Linear(m,m),nn.ReLU(),
        )
        self.block3=nn.Sequential(
            nn.Linear(m,m),nn.ReLU(),
            nn.Linear(m,m),nn.ReLU(),
        )
        self.block4=nn.Sequential(
            nn.Linear(m,m),nn.ReLU(),
            nn.Linear(m,m),nn.ReLU(),
        )
        self.block5=nn.Sequential(
            nn.Linear(m,m),nn.ReLU(),
            nn.Linear(m,m),nn.ReLU(),
        )
        self.block6=nn.Sequential(
            nn.Linear(m,m),nn.ReLU(),
            nn.Linear(m,m),nn.ReLU(),
        )
        self.block7=nn.Sequential(
            nn.Linear(m,m),nn.ReLU(),
            nn.Linear(m,m),nn.ReLU(),
        )
        self.block8=nn.Sequential(
            nn.Linear(m,m),nn.ReLU(),
            nn.Linear(m,m),nn.ReLU(),
        )
        self.out = nn.Linear(m,out)
    def forward(self, x):
        x = self.input(x)
        x = self.block1(x) + x
        x = self.block2(x) + x
        x = self.block3(x) + x
        x = self.block4(x) + x
        x = self.block5(x) + x
        x = self.block6(x) + x
        x = self.block7(x) + x
        x = self.block8(x) + x
        x = self.out(x)
        return x
