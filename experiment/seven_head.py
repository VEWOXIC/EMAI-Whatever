from model.simple_trans import simple
import torch
import numpy as np
from loader.dataloader import day_set
from torch.utils.data import DataLoader
from torch import nn
#import pandas as pd
from sklearn.model_selection import train_test_split

import argparse 
def parse_args():
    parser = argparse.ArgumentParser(description='emai')
    parser.add_argument('--model-name', type=str, default='', help='model name') 
    parser.add_argument('--hidden', type=int, default=40, help='hidden size of variates')
    parser.add_argument('--head', type=int, default=4, help='head size of variates')
    parser.add_argument('--dropout', type=float, default=0.8, help='dropout of hidden layer')

    args = parser.parse_args()
    return args

args = parse_args()

class experiment(object):
    def __init__(self):
        super().__init__()
        self.lr=0.0005
        self.batch_size=7 
        self.epochs=1000
        self.model = self._build_model().cuda()


    def _build_model(self):
        model = simple(args, 3,5)
        return model

    def get_index(self,years,dayofweek,dates):
            day_list = [[] for i in range(7)]
            for day in [0,1,2,3,4,5,6]:
                for i,(year,dow,date) in enumerate(zip(years,dayofweek,dates)):
                    # for data 1, data 2
                    holiday = (int(year[0][:4]) == 2020 and int(date[0]) in [1,25,27,28,95,101,102,104,121,122,177,183,275,276,360,361]) or  (int(year[0][:4]) ==2021 and int(date[0]) in [1,44,46,92,93,95,96,121,122,139,274,287,359,361])
                    # if you use data 3
                    #holiday = (int(year[0][:4]) == 2020 and int(date[0]) in [1,25,27,28,95,101,102,104,121,122,177,184,275,276,361,362]) or (int(year[0][:4]) ==2021 and int(date[0]) in [1,43,44,46,92,93,95,96,121,122,139,165,183,265,274,287,359,361])
                    if day in [0,1,2,3,4,5]:
                        if int(dow[0]) == day and not holiday:
                            day_list[day].append(i)
                    else:
                        if int(dow[0]) == 6 or holiday:
                            day_list[day].append(i)
            return day_list

    def _get_data(self,tt_ratio):
        # data 1: no outliner, no wrong holiday 
        input=np.load('data/input_no_outliner_fix.npy',allow_pickle=True)
        output=np.load('data/output_no_outliner_fix.npy',allow_pickle=True)

        # data 2: has outliner, no wrong holiday
        input=np.load('data/input_fix.npy',allow_pickle=True)
        output=np.load('data/output_fix.npy',allow_pickle=True)

        # data 3: has outliner, wrong holiday 
        input=np.load('data/input_18month_imputed_18prototype.npy',allow_pickle=True)
        output=np.load('./data/output_18month_imputed.npy',allow_pickle=True)
        
        day_list = self.get_index(input[:,:,0],input[:,:,9],input[:,:,10])
        input = input[day_list[self.day]]
        input=input[:,:,[4,5,6,7,8,9,10]].astype(np.float)
        input[:,:,4]=input[:,:,4]/50
        output = output[day_list[self.day]]

        self.bias=output.min()
        self.std=output.std()
        self.out_scale = 1

        train_input,test_input,train_output,test_output=train_test_split(input,output,test_size=tt_ratio,random_state=114514)
        l=input.shape[0]
        self.train_set=day_set(train_input,train_output)
        self.test_set=day_set(test_input,test_output)
        self.train_loader=DataLoader(self.train_set,batch_size=self.batch_size,shuffle=True)
        self.test_loader=DataLoader(self.test_set,batch_size=self.batch_size,shuffle=False)

        return self.train_loader,self.test_loader

    def _get_optim(self):
        return torch.optim.Adam(params=self.model.parameters(),lr=self.lr)
    
    def train(self):
        for day in [0,1,2,3,4,5,6]:
            self.day = day
            self._get_data(0.1)
            self.train_a_day()
            self.validate_a_day()

    def train_a_day(self):
        
        # Train a head
        my_optim=self._get_optim()
        bestloss=1000000
        lossf=nn.MSELoss().cuda()
        l1=nn.L1Loss().cuda()
        for epoch in range(self.epochs):
            self.model.train()
            t_loss=0
            if epoch%100==0 & epoch!=0:
                my_optim.lr=my_optim.lr/2 

            for i,(input,target) in enumerate(self.train_loader):
                input=input.cuda() #[b,t,c]
                input=input[:,:,:5]
                input=input.permute(0,2,1) #[b,c,t]
                target=target.cuda()
                self.model.zero_grad()
                fore=self.model(input)
                fore=fore.squeeze()
                target=target.squeeze()
                loss=torch.sqrt(lossf(fore,target))
                loss.backward()
                my_optim.step()
                t_loss+=loss

            print('Epoch:'+str(epoch)+' loss: '+str(t_loss.item()*self.out_scale/(i+1)))

            with torch.no_grad():
                self.model.eval()
                t_loss=0
                t_l1=0
                for i,(input,target) in enumerate(self.test_loader):
                    input=input.cuda()
                    input=input[:,:,:5]
                    input=input.permute(0,2,1) #[b,c,t]
                    target=target.cuda()
                    fore=self.model(input)
                    fore=fore.squeeze()
                    target=target.squeeze()
                    loss=torch.sqrt(lossf(fore,target))
                    t_loss+=loss
                    t_l1=t_l1+l1(fore,target)

                print('Test loss: '+str(t_loss.item()*self.out_scale/(i+1))+'l1: '+str(t_l1.item()*self.out_scale/(i+1)))
                if t_loss/(i+1)<bestloss:
                    bestloss=t_loss/(i+1)
                    print('get best loss as:',bestloss)
                    torch.save(self.model.state_dict(),'./checkpoints/1day{}.{}'.format(self.day,args.model_name))

    def validate_a_day(self):
        # Validate one head
        self.model.load_state_dict(torch.load('./checkpoints/1day{}.{}'.format(self.day,args.model_name)))
        lossf=nn.MSELoss().cuda()
        l1=nn.L1Loss().cuda()

        with torch.no_grad():
            self.model.eval()
            t_loss=0
            t_l1=0

            out_mse = []
            for i,(input,target) in enumerate(self.test_loader):
                raw_input=input.cuda()
                input=raw_input[:,:,:5]
                info=raw_input[:,0,[5,6]]
                target=target.cuda()
                input=input.permute(0,2,1)
                fore=self.model(input)
                fore=fore.squeeze()
                target=target.squeeze()
                loss=torch.sqrt(lossf(fore,target))
                t_loss+=loss
                out_mse.append(loss.cpu())
                t_l1+=l1(fore,target)
            out_mse = np.array(out_mse)
            print('Day:%d  '%self.day,'batch:',i+1, 'Test mse: '+str(t_loss.item()*self.out_scale/(i+1))+' mae: '+str(t_l1.item()*self.out_scale/(i+1)))

        return i+1, t_loss, t_l1

    def validate(self):
        models = [simple(args, 3,5) for i in range(7)]
        batch_total, mse_total, l1_total = 0,0,0
        # Average validation result
        for i in range(7):
            models[i].load_state_dict(torch.load('./checkpoints/1day{}.{}'.format(i,args.model_name)))
            models[i].cuda()
            self.model = models[i]
            self.day = i
            self._get_data(0.1)
            batch, mse, l1 = self.validate_a_day()
            batch_total+=batch
            mse_total+=mse
            l1_total+=l1

        print('Batch:',batch_total, 'Test mse: '+str(mse_total.item()*self.out_scale/batch_total)+' mae: '+str(l1_total.item()*self.out_scale/batch_total))

    def test(self):
        models = [simple(args, 3,5) for i in range(7)]
        for i in range(7):
            models[i].load_state_dict(torch.load('./checkpoints/day{}.{}'.format(i,args.model_name)))
            models[i].cuda()
        test_set=np.load('./data/test_model.npy',allow_pickle=True)

        # New data
        day_list = self.get_index(test_set[:,:,0],test_set[:,:,7],test_set[:,:,6])
        test_input=test_set[:,:,1:6].astype(float)

        # Old data
        # day_list = self.get_index(test_set[:,:,0],test_set[:,:,9],test_set[:,:,10])
        # test_input=test_set[:,:,4:9].astype(np.float)

        input=torch.from_numpy(test_input).type(torch.float).cuda()
        input[:,:,4]=input[:,:,4]/50
        input=input.permute(0,2,1)
        fore_output=models[0](input)
        fore_output=fore_output.squeeze()
        for i in range(7):
            if len(day_list[i]) != 0:
                fore=models[i](input[day_list[i]])
                fore=fore.squeeze()
                fore_output[day_list[i]] = fore

        print(fore_output)





