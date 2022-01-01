#import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

def PE(t,feature_size):
    cos_PE=np.cos(2*np.pi/feature_size*t)
    sin_PE=np.sin(2*np.pi/feature_size*t)
    return torch.cat((cos_PE,sin_PE),dim=-1)

class weekofyear_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Conv1d(52,1,kernel_size=1)
        
    def forward(self,dayofyear):
        return self.embedding(dayofyear)

class dayofweek_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Conv1d(7,1,kernel_size=1)
    def forward(self,dayofweek):
        return self.embedding(dayofweek)

class hourofday_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Conv1d(24,1,kernel_size=1)
    def forward(self,hourofday):
        return self.embedding(hourofday)

class other_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_embedding=nn.Conv1d(4,128,kernel_size=1)
        self.time_embedding=nn.Conv1d(4,1,kernel_size=1)
        self.FC=nn.Sequential(*[nn.Conv1d(128,128,kernel_size=1)]*5)
        self.relu=nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.output=nn.Conv1d(128,1,kernel_size=1)
    def forward(self,x):
        x=self.channel_embedding(x) 

        x=x.permute(0,2,1)

        x=self.time_embedding(x)

        x=x.permute(0,2,1)

        x=self.FC(x)
        x=self.relu(x)
        x=self.output(x)
        return x

class multiscale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weekofyear_block=weekofyear_block()
        self.dayofweek_block=dayofweek_block()
        self.hourofday_block=hourofday_block()
        self.other_block=other_block()

        self.embedding=nn.Conv1d(3,128,kernel_size=1)

        self.FC=nn.Sequential(*[nn.Conv1d(128,128,kernel_size=1)]*5)
        self.relu=nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.sm=nn.Tanh()


    def forward(self,hod,dow,woy,co): # x [b,c,t]
        W=self.weekofyear_block(woy)
        #W=self.sm(W)
        D=self.dayofweek_block(dow)
        D=self.sm(D)
        H=self.hourofday_block(hod)
        H=self.sm(H)
        C=self.other_block(co)

        return W*D*H+C

        
