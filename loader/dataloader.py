import pandas as pd
import numpy as np

import torch
import torch.utils.data as torch_data
from torch.nn import functional as F

# dayoftheyear and dayofweek hourofday 使用正弦周期函数??? 因为一年365天与第一天实际距离很近


class h_set(torch_data.Dataset):
    def __init__(self, input, output) -> None:
        super().__init__()
        self.input=input
        self.output=output
    def __getitem__(self,index):
        x=torch.from_numpy(self.input[index]).type(torch.float)

        y=torch.from_numpy(np.asarray([self.output[index].mean()])).type(torch.float)

        return x,y
    def __len__(self):
        return self.input.shape[0]




class day_set(torch_data.Dataset):
    def __init__(self, input, output) -> None:
        super().__init__()
        self.input=input
        self.output=output
    def __getitem__(self,index):


        x=self.input[index].astype(np.float)
        # if x[:4].any()==1:
        #     for i in range(9*4,16*4):
        #         x[i][4]=1


        #x=torch.from_numpy(self.input[index].astype(np.float)).type(torch.float)
        x=torch.from_numpy(x).type(torch.float)
        
        y=torch.from_numpy(self.output[index].astype(np.float)).type(torch.float)

        return x,y
    def __len__(self):
        return self.input.shape[0] 

class day_info_set(torch_data.Dataset):
    def __init__(self, input, output) -> None:
        super().__init__()
        self.input=input
        self.output=output
    def __getitem__(self,index):


        x=self.input[index,:,[4,5,6,7,8]].astype(np.float)
        # if x[:4].any()==1:
        #     for i in range(9*4,16*4):
        #         x[i][4]=1
        info=self.input[index,0,[0,9,10]].tolist()


        #x=torch.from_numpy(self.input[index].astype(np.float)).type(torch.float)
        x[:,4]=x[:,4]*100
        x=torch.from_numpy(x).type(torch.float)
        
        y=torch.from_numpy(self.output[index].astype(np.float)).type(torch.float)

        return x,y,info
    def __len__(self):
        return self.input.shape[0] 

if __name__ == '__main__':
    input=np.load('./data/input_18month_no_nan.npy',allow_pickle=True)

    output=np.load('./data/output_18month_no_nan.npy',allow_pickle=True)
    aset=day_set(input,output)
    print(aset[1])
