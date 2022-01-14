#import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class self_attention(nn.Module):
    def __init__(self, hidden, head):
        super().__init__()
        
        self.attention=nn.MultiheadAttention(hidden,head)
        self.ln=nn.LayerNorm([hidden,96])
    def forward(self,x):
        x_res=x
        x=self.attention(x,x,x)[0]+x_res #å·²è¯å®ž æ— residual l1=1800
        x=x.permute(1,2,0)#[b,c,t]

        x=self.ln(x)
        x=x.permute(2,0,1)
        return x

class simple(nn.Module):
    def __init__(self,args, attention_layer=2,channel_expand=False): # 3å±‚l1 1000+ 7å±‚1400 æ›´å¤šæ•°æ®7å±‚ 697 æ›´å¤§lrå¯åˆ°505
        super().__init__()
        self.channel_expand=channel_expand
        self.hidden = args.hidden
        self.attention=nn.Sequential(*[self_attention(self.hidden, args.head)]*attention_layer)
        self.channel_FC=nn.Conv1d(self.hidden, 1,kernel_size=1) #k=3 l1 1400

        if channel_expand:
            self.channel_expand=nn.Conv1d(self.hidden,128,kernel_size=1)
            self.channel_FC=nn.Sequential(*[nn.Conv1d(128,128,kernel_size=1)]*channel_expand)
            
            self.dropout=nn.Dropout(args.dropout)
            self.channel_shrink=nn.Conv1d(128,1,kernel_size=1)

        self.time_FC=nn.Conv1d(96,24,kernel_size=1,padding=0) # k=3 l1 700~ k=5 l1 900~ å¢žè¡¥æ•°æ® k=1 Test loss: tensor(619.6642, device='cuda:0')l1: tensor(446.2283, device='cuda:0') k=3 Test loss: tensor(688.7370, device='cuda:0')l1: tensor(492.4089, device='cuda:0')
        self.relu = nn.GELU()

        self.fc = nn.Linear(5, self.hidden)
        
    def forward(self,x): # x [b,c,t]
        x = self.fc(x.permute(0,2,1)).permute(0,2,1)
        x=x.permute(2,0,1) # [t,b,c]
        x=self.attention(x) # [t,b,c]
        x=x.permute(1,2,0) # [b,c,t]
        
        if self.channel_expand:
            x=self.channel_expand(x)
            x=self.channel_FC(x)
            x=self.dropout(x)
            x=self.relu(x)
            x=self.channel_shrink(x)

        else:
            x=self.channel_FC(x) # [b,c1,t]
        x=x.permute(0,2,1)
        
        x=self.time_FC(x) # [b,t,c]
        return x.permute(0,2,1) #[b,c,t]

class two_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.weekday_head=simple()
        self.weekend_head=simple()
    def forward(self,x,dayofweek):
        pass


if __name__ == '__main__':
    m=simple()
    b,t,c=3,96,5
    x=torch.rand((b,c,t))
    
    y=m(x)
    print(x.shape)
    print(y.shape)