from model.simple_trans import simple
import torch
import numpy as np
from loader.dataloader import day_set
from torch.utils.data import DataLoader
from torch import nn
#import pandas as pd

class experiment(object):
    def __init__(self) -> None:
        super().__init__()
        self.lr=0.0001
        self.batch_size=24
        self.epochs=500
        self.model = self._build_model().cuda()
        self._get_data(0.7)

    def _build_model(self):
        model=simple()
        return model
    
    def _get_data(self,tt_ratio):
        input=np.load('./data/input_18month_no_nan_weekend.npy',allow_pickle=True)

        output=np.load('./data/output_18month_no_nan.npy',allow_pickle=True)
        l=input.shape[0]
        train_input=input[:int(l*tt_ratio)]
        train_output=output[:int(l*tt_ratio)]
        test_input=input[int(l*tt_ratio):]
        test_output=output[int(l*tt_ratio):]
        self.train_set=day_set(train_input,train_output)
        self.test_set=day_set(test_input,test_output)
        self.train_loader=DataLoader(self.train_set,batch_size=self.batch_size,shuffle=True)
        self.test_loader=DataLoader(self.test_set,batch_size=self.batch_size,shuffle=False)

        return self.train_loader,self.test_loader

    def _get_optim(self):
        return torch.optim.Adam(params=self.model.parameters(),lr=self.lr)

    
    def train(self):
        my_optim=self._get_optim()
        bestloss=1000000
        
        lossf=nn.MSELoss().cuda()
        l1=nn.L1Loss().cuda()

        for epoch in range(self.epochs):
            t_loss=0
            for i,(input,target) in enumerate(self.train_loader):
                input=input.cuda() #[b,t,c]
                #print(input.shape)
                input=input.permute(0,2,1) #[b,c,t]
                #print(input.shape)
                target=target.cuda()
                self.model.zero_grad()

                fore=self.model(input)
                fore=fore.squeeze()
                target=target.squeeze()
                #print(fore,target)
                loss=torch.sqrt(lossf(fore,target))

                loss.backward()
                my_optim.step()

                t_loss+=loss
                #print(loss)

            print('Epoch:'+str(epoch)+' loss: '+str(t_loss/i))
            if t_loss/i<bestloss:
                bestloss=t_loss/i
                torch.save(self.model.state_dict(),'./checkpoints/simple_trans_weekend.model')

            with torch.no_grad():
                t_loss=0
                t_l1=0
                for i,(input,target) in enumerate(self.test_loader):
                    input=input.cuda()
                    input=input.permute(0,2,1) #[b,c,t]
                    target=target.cuda()
                    fore=self.model(input)
                    fore=fore.squeeze()
                    target=target.squeeze()
                    loss=torch.sqrt(lossf(fore,target))
                    t_loss+=loss
                    t_l1=t_l1+l1(fore,target)
                print('Test loss: '+str(t_loss/i)+'L1: '+str(t_l1/i))

    def test(self):
        self.model.load_state_dict(torch.load('./checkpoints/simple_trans_weekend.model'))
        lossf=nn.L1Loss().cuda()
        with torch.no_grad():
            t_loss=0
            for i,(input,target) in enumerate(self.test_loader):
                input=input.cuda()
                print(input[2,:,-1].squeeze())
                target=target.cuda()
                input=input.permute(0,2,1) #[b,c,t]
                fore=self.model(input)
                fore=fore.squeeze()
                target=target.squeeze()
                loss=torch.sqrt(lossf(fore,target))
                t_loss+=loss
                print(target,fore)
                #print(fore, target)
            print('Test loss: '+str(t_loss/i))



            



if __name__=='__main__':

    model=simple()
    dummyinput=torch.FloatTensor(1,5,4)
    print(dummyinput)


