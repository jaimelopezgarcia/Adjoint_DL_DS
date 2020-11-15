import torch 
import numpy as np
from functools import partial
from torch import nn
from tqdm import tqdm
import os


act1 = lambda x:x
act2 = torch.sin

class Encoder(nn.Module):
    def __init__(self,in_features, hidden_features, out_features):
        super().__init__()
        
        self._l1 = nn.Linear(in_features, out_features)
        
        self._layers = [self._l1]
        
    def forward(self,x):
        
        x = act1( self._layers[0](x) )
        
        for layer in self._layers[1:]:
            x = act1( layer(x) )
        
        return x
    
class EncoderVariational(nn.Module):
    
    def __init__(self,in_features, hidden_features, out_features):
        super().__init__()
        
        self._l1 = nn.Linear(in_features, hidden_features)

        self._mean = nn.Linear(hidden_features, out_features)
        self._logvar = nn.Linear(hidden_features, out_features)
        
        self._layers = [self._l1, self._l2, self._l3]
        
    def forward(self,x):
        
        x = act2( self._layers[0](x) )
        
        for layer in self._layers[1:]:
            x = act2( layer(x) )
            
        mean = self._mean(x)
        log_var = self._logvar(x)
        stddev = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(mean)
        z = mean + stddev * epsilon
        
        return z, mean, log_var
    
    
    
class TransformNet(nn.Module):
    
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self._l1 = nn.Linear(in_features, hidden_features)
        self._q = nn.Linear(hidden_features, out_features)
        self._p = nn.Linear(hidden_features, out_features)
        
        self._layers = [self._l1]
    
    def forward(self, x):
       
        x = act1( self._layers[0](x) )
        
        for layer in self._layers[1:]:
            x = act1( layer(x) )
            
        q = self._q(x) 
        p =  self._p(x) 
        
        return q,p
    
class HamiltonianNet(nn.Module):
    
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self._l1 = nn.Linear(in_features, hidden_features)
        self._l2 = nn.Linear(hidden_features, hidden_features)
        self._l3 = nn.Linear(hidden_features, hidden_features)
        self._l4 = nn.Linear(hidden_features, hidden_features)
        self._h = nn.Linear(hidden_features, 1)

        self._layers = [self._l1, self._l2, self._l3, self._l4]
        
        self.w0 = 5.
    def forward(self, q, p):
        
        x = torch.cat([q,p],dim = 1)
       
        x = act2(self.w0*self._layers[0](x) )
        
        for layer in self._layers[1:]:
            x = act2( layer(x) )
            
        h = self._h(x)
        
        return h
    
    
    
class Decoder(nn.Module):
    def __init__(self,in_features, hidden_features, out_features):
        super().__init__()
        
        self._l1 = nn.Linear(in_features, hidden_features)

        self._out = nn.Linear(hidden_features, out_features)
        
        self._layers = [self._l1]
        
    def forward(self,q):
                
        x = act1( self._layers[0](q) )
        
        for layer in self._layers[1:]:
            x = act1( layer(x) )
        
        x = self._out(x)
        
        return x   
    

def get_grads( q, p, hnn):

        H = hnn(q=q, p=p)

        dq_dt = torch.autograd.grad(H,
                                    p,
                                    create_graph=True,
                                    retain_graph=True,
                                    grad_outputs=torch.ones_like(H))[0]

        dp_dt = -torch.autograd.grad(H,
                                     q,
                                     create_graph=True,
                                     retain_graph=True,
                                     grad_outputs=torch.ones_like(H))[0]



        return dq_dt, dp_dt, H.detach().cpu().numpy()
    
    
def euler_step(q, p, hnn, dt):

    dq_dt, dp_dt, energy = get_grads(q, p, hnn)

    q_next = q + dt * dq_dt
    p_next = p + dt * dp_dt
    
    return q_next, p_next, energy



def lf_step(q, p, hnn,dt):

    _, dp_dt,energy = get_grads(q, p, hnn)
    
    p_next_half = p + dp_dt * (dt) / 2
    q_next = q + p_next_half * dt
    _, dp_next_dt,energy = get_grads(q_next, p_next_half, hnn)
    p_next = p_next_half + dp_next_dt * (dt) / 2
    
    return q_next, p_next ,energy





def calculate_trajectory(x, n_steps,step, 
                        encoder,
                        tnet,
                        hnn,
                        variational = False):
    if not(variational):
        z = encoder(x)
    else:
        z,mean,logvar = encoder(x)
        
    q,p = tnet(z)
        
    qs,ps = [], []
    energies = []
    
    qnext,pnext, energy = lf_step(q,p,hnn,step)   
    qs.append(qnext.unsqueeze(dim=1))
    ps.append(pnext.unsqueeze(dim=1))
    energies.append(energy)
    
    for i in range(n_steps-1):
        qnext,pnext,energy = lf_step(qnext,pnext,hnn,step)
        qs.append(qnext.unsqueeze(dim = 1))
        ps.append(pnext.unsqueeze(dim = 1))
        energies.append(energy)
    qs = torch.cat(qs, dim = 1)
    ps = torch.cat(ps,dim = 1)
    
    if not(variational):
        return qs,ps, energies
    else:
        return qs,ps,energies,mean,logvar