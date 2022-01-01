#import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class self_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention=nn.MultiheadAttention(5,5)
    def forward(self,x):
        x=self.attention(x,x,x)
        return x[0]

class simple(nn.Module):
    def __init__(self,attention_layer=3):
        super().__init__()

        self.attention=nn.Sequential(*[self_attention()]*attention_layer)
        self.channel_FC=nn.Conv1d(5,1,kernel_size=1)
        self.time_FC=nn.Conv1d(96,24,kernel_size=1)
        

    def forward(self,x): # x [b,c,t]
        x=x.permute(2,0,1) # [t,b,c]
        x=self.attention(x) # [t,b,c]
        x=x.permute(1,2,0) # [b,c,t]
        x=self.channel_FC(x) # [b,c1,t]
        x=x.permute(0,2,1)
        x=self.time_FC(x) # [b,t,c]

        return x.permute(0,2,1) #[b,c,t]

if __name__ == '__main__':
    m=simple()
    b,t,c=3,96,5
    x=torch.rand((b,c,t))
    
    y=m(x)
    print(x.shape)
    print(y.shape)
