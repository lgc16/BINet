import torch
import math
import numpy as np
import scipy.special as scp

def G_bd(domain_type,problem_type,layer,sample_x,h,normal,in_or_out):
    sample_num = len(sample_x)
    vertex_num = len(h)
    M = int(sample_num/vertex_num)
    j0 = [math.floor(j/M) for j in range(sample_num) ]
    j1 = [math.floor((j-1)/M) for j in range(sample_num) ]
    if problem_type =='Laplace':
        G = torch.zeros(sample_num,sample_num)
        if domain_type == 'polygon':
            for i in range(sample_num):
                if (i%M != 0):#and (i%M != 1) and (i%M != M-1):
                    if layer == 2:
                        r = sample_x - sample_x[i,:]
                        d = r.norm(dim=1)
                        G[i,:] = -1/(2*np.pi)*(r*normal[j0]*h[j0]+r*normal[j1]*h[j1]).sum(axis=1)/(2*d*d)
                        if in_or_out=='in':
                            G[i,i] = -1/2
                        else:
                            G[i,i] = 1/2
        elif domain_type == 'smoothdomain':
            return 0
        return G
    elif problem_type[0:9] == 'Helmholtz':
        k = float(problem_type[9:len(problem_type)])
        G_r, G_i = torch.zeros(sample_num,sample_num), torch.zeros(sample_num,sample_num)
        if domain_type=='smoothdomain':
            KaparR2 = torch.ones(sample_num)
            KaparR2[1],KaparR2[2] =KaparR2[1] + 1.825748064736159, KaparR2[2] - 1.325748064736159
            j_index = np.array([i for i in range(sample_num)])

            for i in range(sample_num):
                if layer == 1:
                    r0 = (sample_x-sample_x[i,:]).norm(dim=1)
                    g = scp.hankel1(0,k*r0)
                    c = KaparR2[abs((j_index-i+round(sample_num/2))%sample_num-round(sample_num/2))]*(h.norm(dim=1))/sample_num*2*np.pi
                    G_r[i,:], G_i[i,:] = -c*(-g.imag/4), -c*(g.real/4)
                    G_r[i,i], G_i[i,i] = 0, 0
                elif layer == 2:
                    r0 = sample_x-sample_x[i,:]
                    d = r0.norm(dim=1)
                    n1 = (normal*r0).sum(axis=1)
                    c = KaparR2[abs((j_index-i+round(sample_num/2))%sample_num-round(sample_num/2))]*(h.norm(dim=1))/sample_num*2*np.pi
                    G_r[i,:] = scp.hankel1(1,k*d).imag/4*n1/d*k*(-c)
                    G_i[i,:] = -scp.hankel1(1,k*d).real/4*n1/d*k*(-c)
                    if in_or_out=='out':
                        G_r[i,i],G_i[i,i] = -1/2,0
            return G_r, G_i 


def G_in(problem_type,layer,x_in,sample_x,h,normal):
    sample_num = len(sample_x)
    vertex_num = len(h)
    sample = len(x_in)
    M = int(sample_num/vertex_num)
    j0 = [math.floor(j/M) for j in range(sample_num) ]
    j1 = [math.floor((j-1)/M) for j in range(sample_num) ]
    if problem_type =='Laplace':
        G = torch.zeros(sample,sample_num)
        for i in range(sample):
            if layer==2:
                r = sample_x-x_in[i,:]
                d = r.norm(dim=1)
                G[i,:] = -1/(2*np.pi)*(r*normal[j0]*h[j0]+r*normal[j1]*h[j1]).sum(axis=1)/(2*d*d)
        return G
    elif problem_type[0:9] == 'Helmholtz':
        k = float(problem_type[9:len(problem_type)])
        G_r, G_i = torch.zeros(sample,sample_num), torch.zeros(sample,sample_num)
        for i in range(sample):
            d = sample_x-x_in[i,:]
            r0 = d.norm(dim=1)
            c_in = (h.norm(dim=1))/sample_num*2*np.pi
            G_r[i,:] = -scp.hankel1(0,k*r0).imag/4*c_in
            G_i[i,:] = scp.hankel1(0,k*r0).real/4*c_in
        return G_r,G_i

                    