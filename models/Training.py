import torch
import time
import sys
sys.path.append('../Models')
sys.path.append('../Utils')
import GenerateIntegralMatrix as IM
import BoundaryCondition as Bdc

def laplace_training(net,sample_x,sample_u,G,loss_func,optimizer,Epoch,each_epoch):
    loss_all = torch.zeros(Epoch+1)
    time0 = time.time()
    for epoch in range(Epoch+1):
        sample_h = net(sample_x)
        
        u0 = (G@sample_h)

        loss = loss_func(sample_u,u0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all[epoch] = loss.detach().cpu()
        if epoch%each_epoch==0:
            print('loss, epoch, computation time:','%.6f'%loss.detach().numpy(),epoch,'%.4f'%(time.time()-time0))
            time0 = time.time()
    return net,loss_all

def Helmholtz_training(net,sample_x,sample_u_r,sample_u_i,G_r,G_i,loss_func,optimizer,Epoch,each_epoch):
    loss_all = torch.zeros(Epoch+1)
    time0 = time.time()
    for epoch in range(Epoch+1):
        
        sample_h_r,sample_h_i = net(sample_x)[:,0],net(sample_x)[:,1]
    
        u0_r = (G_r@sample_h_r-G_i@sample_h_i)
        u0_i = (G_r@sample_h_i+G_i@sample_h_r)
        
        loss = loss_func(sample_u_r,sample_u_i,u0_r,u0_i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all[epoch] = loss.detach().cpu()
        if epoch%each_epoch==0:
            print('loss, epoch, computation time:','%.6f'%loss.detach().numpy(),epoch,'%.4f'%(time.time()-time0))
            time0 = time.time()
    return net,loss_all

def Helmholtz_training_operator(net,k,sample_xk,sample_u_r,sample_u_i,G_r,G_i,loss_func,optimizer,Epoch,each_epoch):
    loss_all = torch.zeros(Epoch+1)
    k_num = len(k)
    time0 = time.time()
    for epoch in range(Epoch+1): 
        loss = 0
        for l in range(k_num):
            sample_h = net(sample_xk[l])
            sample_h_r,sample_h_i = sample_h[:,0],sample_h[:,1]
            u0_r = (G_r[l]@sample_h_r-G_i[l]@sample_h_i)
            u0_i = (G_r[l]@sample_h_i+G_i[l]@sample_h_r)
            loss = loss + loss_func(sample_u_r[l,:],sample_u_i[l,:],u0_r,u0_i)*k[l]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all[epoch] = loss.cpu().detach()
        if epoch%each_epoch==0:
            print('loss, epoch, computation time:','%.4f'%loss.detach().numpy(),epoch,'%.4f'%(time.time()-time0))
            time0 = time.time()
    return net,loss_all


