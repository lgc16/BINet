import torch
import scipy.special as scp
import numpy as np
def exact_sol(a,b,example):
    if example == 'ex1-1':
        k = 4
        return torch.exp(k*a)*torch.sin(k*b) 
    elif example == 'ex1-2':
        k = 8
        return torch.exp(k*a)*torch.sin(k*b) 
    elif example == 'ex2':
        return 1-abs(b)+1-abs(a)             
    elif example == 'ex3-1':
        k , theta0 = 1 , np.pi/7
        k1,k2 = k*np.cos(theta0) , k*np.sin(theta0) 
        return torch.cos(k1*a+k2*b), torch.sin(k1*a+k2*b)
    elif example == 'ex3-2':
        k , theta0 = 4 , np.pi/7
        k1,k2 = k*np.cos(theta0) , k*np.sin(theta0) 
        return torch.cos(k1*a+k2*b), torch.sin(k1*a+k2*b)

def exact_sol_operator(sample_x,parameter,example):
    if example == 'ex4-1':
        l = len(parameter)
        sample_num = len(sample_x)
        sample_xk = torch.zeros(l,sample_num,3)

        sample_u_r = torch.zeros(l,sample_num)
        sample_u_i = torch.zeros(l,sample_num)
        for i in range(l):
            sample_u_r[i,:] = scp.hankel1(0,parameter[i]*torch.sqrt((sample_x[:,0])**2+sample_x[:,1]**2)).real
            sample_u_i[i,:] = scp.hankel1(0,parameter[i]*torch.sqrt((sample_x[:,0])**2+sample_x[:,1]**2)).imag
            sample_xk[i,:,0:2] = sample_x
            sample_xk[i,:,2] = parameter[i]
        return sample_u_r,sample_u_i,sample_xk

    