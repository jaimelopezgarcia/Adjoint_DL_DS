import torch 
import numpy as np
from torch import nn
from torchdiffeq import odeint, odeint_adjoint



class OdeFun(torch.nn.Module):
    
    def __init__(self,width,augment_dim,in_dim):
        
        super().__init__()
        
        self.layer1 = torch.nn.Linear(in_dim+augment_dim,width)
        self.layer2 = torch.nn.Linear(width,width)
        self.layer3 = torch.nn.Linear(width,in_dim+augment_dim)
        self.augment_dim = augment_dim
        
    def forward(self,t,X):
        
        X = torch.relu(self.layer1(X))
        X = torch.relu(self.layer2(X))
        X = self.layer3(X)
        
        return X

class OdeBlock(nn.Module):

    def __init__(self, odefun, tol=1e-3):
        
        super().__init__()
        self.odefun = odefun
        self.tol = tol
        
    def forward(self, x, eval_times=None, only_last = True):

        if eval_times is None:

            #integration_time = torch.tensor([0, 1]).float().type_as(x)
            integration_time = torch.arange(0,1,0.1).float().type_as(x)

        else:
            
            integration_time = eval_times.type_as(x)


        if self.odefun.augment_dim > 0:

                aug = torch.zeros(x.shape[0], self.odefun.augment_dim)
                x_aug = torch.cat([x, aug], 1)
        else:
            x_aug = x

        out = odeint_adjoint(self.odefun, x_aug, integration_time,
                             rtol=self.tol, atol=self.tol, method='dopri5',
                             options={'max_num_steps': 1000})

        if only_last:
            return out[-1]
        else:
            return out
        
class OdeNet(nn.Module):
    
    def __init__(self, width, augment_dim, in_dim, out_dim):
        
        super().__init__()
        self.odefun = OdeFun(width,augment_dim,in_dim)
        self.odeblock = OdeBlock(self.odefun)
        self.linear_out = nn.Linear(in_dim+augment_dim,out_dim)
    
    def forward(self,x,eval_times = None, only_last = True):
        
        x = self.odeblock(x,eval_times = eval_times, only_last = only_last )
        
        out = self.linear_out(x)
        
        return out
    
    
    
    
    
class ConvODEFunc(nn.Module):

    def __init__(self, img_size=(1,28,28), num_filters=20, augment_dim=30):
        super().__init__()
        self.augment_dim = augment_dim
        self.img_size = img_size
        self.channels, self.height, self.width = img_size
        self.channels += augment_dim
        self.num_filters = num_filters

        self.conv1 = nn.Conv2d(self.channels, self.num_filters,
                               kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters,
                               kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_filters, self.channels,
                               kernel_size=1, stride=1, padding=0)

        self.non_linearity = nn.ReLU(inplace=True)


    def forward(self, t, x):

        out = self.conv1(x)
        out = self.non_linearity(out)
        out = self.conv2(out)
        out = self.non_linearity(out)
        out = self.conv3(out)
        return out
    
class ODEBlockConv(nn.Module):

    def __init__(self, odefunc,  tol=1e-3):
        super().__init__()
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None, only_last = True):

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)


        if self.odefunc.augment_dim > 0:
            
            batch_size, channels, height, width = x.shape
            aug = torch.zeros(batch_size, self.odefunc.augment_dim,
                              height, width)
            x_aug = torch.cat([x, aug], 1)

        else:
            x_aug = x


        out = odeint_adjoint(self.odefunc, x_aug, integration_time,
                                 rtol=self.tol, atol=self.tol, method='dopri5',
                                 options={'max_num_steps': 1000})

        if only_last:
            return out[1]  # Return only final time
        else:
            return out



class ConvODENet(nn.Module):

    def __init__(self, img_size=(1,28,28), num_filters = 20, output_dim=10,
                 augment_dim=30 ,
                 tol=1e-3):
        super().__init__()
        self.img_size = img_size
        self.num_filters = num_filters
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.flattened_dim = (img_size[0] + augment_dim) * img_size[1] * img_size[2]
        self.tol = tol

        odefunc = ConvODEFunc( img_size = img_size, num_filters= num_filters, augment_dim= augment_dim )

        self.odeblock = ODEBlockConv( odefunc, tol=tol)

        self.linear_layer = nn.Linear(self.flattened_dim, self.output_dim)

    def forward(self, x, return_features=False, eval_times = None, only_last = True):
        features = self.odeblock(x, eval_times = eval_times)
        pred = self.linear_layer(features.view(features.size(0), -1))
        if return_features:
            return features, pred
        return pred
    
    
    
import pickle
import io
import os

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
        

        
_dir = os.path.dirname(__file__)



def get_convode():
    
    conv = ConvODENet()
    with open(os.path.join(_dir,"save_models/convodenet"),'rb') as f:
        s_dict = contents = CPU_Unpickler(f).load()
        conv.load_state_dict(s_dict)
    return conv


def conv_ode_forward(dt_test,im_number,convode,Tf,step):
    
    block = convode.odeblock
    
    im = dt_test.tensors[0][im_number].unsqueeze(dim = 0)
    
    time = torch.arange(0,Tf,step)
    
    out = block(im, eval_times = time, only_last = False)
    
    return out