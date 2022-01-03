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
        x=torch.from_numpy(self.input[index].astype(np.float)).type(torch.float)
        
        y=torch.from_numpy(self.output[index].astype(np.float)).type(torch.float)

        return x,y
    def __len__(self):
        return self.input.shape[0] 