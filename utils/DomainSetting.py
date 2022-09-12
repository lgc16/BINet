import torch
import numpy as np


def polygondomain(M,vertex):
    line = (torch.linspace(0,1-1/M,M)).reshape(-1,1)
    vertex_num = len(vertex)
    vertex = vertex[[(i%vertex_num) for i in range(vertex_num+1)]]

    rot = torch.Tensor([[0,-1],[1,0]])  # Rotate 90 degrees clockwise

    normal = torch.zeros(vertex_num,2)  
    h = torch.zeros(vertex_num,1)     
    for i in range(vertex_num):
        normal[i,:] = (vertex[i+1,:]-vertex[i,:])@rot
        normal[i,:] = normal[i,:]/(normal[i,:].norm())
        h[i] = (vertex[i+1,:]-vertex[i,:]).norm()/M

    sample_x = torch.zeros(vertex_num*M,2)
    for i in range(vertex_num):
        sample_x[i*M:i*M+M,:] = vertex[i,:]+(vertex[i+1,:]-vertex[i,:])*line
    return sample_x,normal,h

def petaldomain(M):

    theta = torch.linspace(2*np.pi/M,2*np.pi,M).reshape(-1,1)
    r = 9/20-1/9*torch.cos(5*theta)
    x0,x1 = torch.cos(theta),torch.sin(theta)
    sample_x = torch.cat((x0,x1),1)*r


    rprime = torch.cat((-x1*r+5/9*torch.sin(5*theta)*x0,x0*r+5/9*torch.sin(5*theta)*x1),1)
    normal = rprime@torch.Tensor([[0,-1],[1,0]])
    normal = normal/((normal*normal).sum(axis=1).reshape(-1,1).sqrt())
    return sample_x, normal, rprime


