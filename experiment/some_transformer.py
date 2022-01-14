from model.simple_trans import simple
import torch
import numpy as np
from loader.dataloader import day_set
from torch.utils.data import DataLoader
from torch import nn
#import pandas as pd
from sklearn.model_selection import train_test_split

class experiment(object):
    def __init__(self) -> None:
        super().__init__()
        self.lr=0.0005
        self.batch_size=7 # 7 l1loss 524 14 830
        self.epochs=1000
        self.model = self._build_model().cuda()
        self._get_data(0.1,False)

    def _build_model(self):
        model=simple(2) # expand 训练稳定许多 1000epoch 474 false时会卡在600左右
        print(model)
        return model
    
    def _get_data(self,tt_ratio,normalize):
        input=np.load('./data/input_15month_no_nan.npy',allow_pickle=True)

        output=np.load('./data/output_15month_no_nan.npy',allow_pickle=True)
        input[:,:,4]=input[:,:,4]*100
        self.bias=output.min()
        self.std=output.std()
        #output=(output-self.bias)/self.std
        train_input,test_input,train_output,test_output=train_test_split(input,output,test_size=tt_ratio,random_state=114514)
        print(train_input.shape,test_input.shape)
        if normalize:
            new_input=np.zeros(input.shape)
            new_input[:,:,0]=(input[:,:,0]-input[:,:,0].min())/input[:,:,0].std()
            new_input[:,:,1]=(input[:,:,1]-input[:,:,1].min())/input[:,:,1].std()
            new_input[:,:,2]=(input[:,:,2]-input[:,:,2].min())/input[:,:,2].std()
            new_input[:,:,3]=(input[:,:,3]-input[:,:,3].min())/input[:,:,3].std()
            new_output=(output-output.min())/output.std()
            input=new_input
            output=new_output
            print(input.shape)

        l=input.shape[0]
        # train_input=input[:int(l*tt_ratio)]
        # train_output=output[:int(l*tt_ratio)]
        # test_input=input[int(l*tt_ratio):]
        # test_output=output[int(l*tt_ratio):]
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
            self.model.train()
            t_loss=0
            if epoch%100==0 & epoch!=0:
                my_optim.lr=my_optim.lr/2 #j加入动态调整 l1loss 514
            for i,(input,target) in enumerate(self.train_loader):
                input=input.cuda() #[b,t,c]

                print(input.shape)
                input=input.permute(0,2,1) #[b,c,t]
                #input=input[:,[2,3,4],:]
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
            

            with torch.no_grad():
                self.model.eval()
                t_loss=0
                t_l1=0
                for i,(input,target) in enumerate(self.test_loader):
                    
                    input=input.cuda()
                    input=input.permute(0,2,1) #[b,c,t]
                    

                    #input=input[:,[2,3,4],:]

                    target=target.cuda()
                    fore=self.model(input)
                    fore=fore.squeeze()
                    target=target.squeeze()
                    loss=torch.sqrt(lossf(fore,target))
                    t_loss+=loss
                    t_l1=t_l1+l1(fore,target)
                print('Test loss: '+str(t_loss/i)+'L1: '+str(t_l1/i))
                if t_loss/i<bestloss:
                    bestloss=t_loss/i
                    torch.save(self.model.state_dict(),'./checkpoints/simple_trans_15m.model')

    def test(self):
        self.model.load_state_dict(torch.load('./checkpoints/simple_trans_15m_388.model'))
        lossf=nn.MSELoss().cuda()
        l1=nn.L1Loss().cuda()

        with torch.no_grad():
            self.model.eval()
            t_loss=0
            t_l1=0
            weekday_t=[]
            weekend_t=[]
            weekday_f=[]
            weekend_f=[]
            for i,(input,target) in enumerate(self.test_loader):
                
                input=input.cuda()
                
                #print(input[2,:,-1].squeeze())
                target=target.cuda()
                #input[:,:,-1]=torch.zeros((96))+100
                input=input.permute(0,2,1) #[b,c,t]
                print(input.shape)
                #input=input[:,[2,3,4],:]
                fore=self.model(input)
                fore=fore.squeeze()
                target=target.squeeze()
                loss=torch.sqrt(lossf(fore,target))
                t_loss+=loss
                t_l1+=l1(fore,target)
                print(target,fore)
                #print(fore, target)
                for j in range(target.shape[0]):
                    if target.shape[0]==24:
                        break
                    #print(input.shape,target.shape[0])
                    sample=input[j]
                    #print(sample.shape,sample[4,50])
                    if sample[4,50]==100:
                        weekday_t.append(target[j].cpu().numpy().tolist())
                        weekday_f.append(fore[j].cpu().numpy().tolist())
                    else:
                        weekend_t.append(target[j].cpu().numpy().tolist())
                        weekend_f.append(fore[j].cpu().numpy().tolist())
            print('Test loss: '+str(t_loss/i)+'l1: '+str(t_l1/i))
            print('wday_f=')
            print(weekday_f)
            print('wday_t=')
            print(weekday_t)
            print('wend_f=')
            print(weekend_f)
            print('wend_t=')
            print(weekend_t)
        

if __name__=='__main__':

    model=simple()
    dummyinput=torch.FloatTensor(1,5,4)
    print(dummyinput)


