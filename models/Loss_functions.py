import torch
import torch.nn as nn

class Laplace_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,u_exact,u_Green):
        return torch.mean(torch.pow((u_exact-u_Green),2)) 

class Helmholtz_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,u_exact_r,u_exact_i,u_Green_r,u_Green_i):
        return torch.mean(torch.pow((u_exact_r-u_Green_r),2)) +torch.mean(torch.pow((u_exact_i-u_Green_i),2)) 