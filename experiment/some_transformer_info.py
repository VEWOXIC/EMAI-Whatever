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
        model=simple(3,5) # expand 训练稳定许多 1000epoch 474 false时会卡在600左右
        print(model)
        return model
    
    def _get_data(self,tt_ratio,normalize):
        input=np.load('./data/input_18month_imputed_18prototype.npy',allow_pickle=True)
        # input=np.load('./data/input_15month_no_nan_withprototype_holiday.npy',allow_pickle=True)
        input=input[:,:,[4,5,6,7,8,9,10]].astype(np.float)

        output=np.load('./data/output_18month_imputed_18prototype.npy',allow_pickle=True)
        #output=np.load('./data/output_15month_no_nan_withdate.npy',allow_pickle=True)
        input[:,:,4]=input[:,:,4]/50
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
        # train_input=input[:int(l*(1-tt_ratio))]
        # train_output=output[:int(l*(1-tt_ratio))]
        # test_input=input[int(l*(1-tt_ratio)):]
        # test_output=output[int(l*(1-tt_ratio)):]
        # train_input=input[:-90]
        # train_output=output[:-90]
        # test_input=input[1:]
        # test_output=output[1:]
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

                #print(input.shape)
                input=input[:,:,:5]
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
                    input=input[:,:,:5]
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
                    torch.save(self.model.state_dict(),'./checkpoints/simple_trans_18m_newprototype.model')

    def validate(self):
        self.model.load_state_dict(torch.load('./checkpoints/simple_trans_18m_newprototype.model'))
        lossf=nn.MSELoss().cuda()
        l1=nn.L1Loss().cuda()

        with torch.no_grad():
            self.model.eval()
            t_loss=0
            t_l1=0
            w_t=[[],[],[],[],[],[],[]]
            w_f=[[],[],[],[],[],[],[]]
            #####
            e=[]
            d=[]
            #####
            for i,(input,target) in enumerate(self.test_loader):
                
                raw_input=input.cuda()
                input=raw_input[:,:,:5]
                info=raw_input[:,0,[5,6]]
                print(info)
                #print(input[2,:,-1].squeeze())
                target=target.cuda()
                #input[:,:,-1]=torch.zeros((96))+100
                input=input.permute(0,2,1) #[b,c,t]
                #print(input.shape)
                #input=input[:,[2,3,4],:]
                fore=self.model(input)
                fore=fore.squeeze()
                target=target.squeeze()
                loss=torch.sqrt(lossf(fore,target))
                t_loss+=loss
                t_l1+=l1(fore,target)
                #print(target,fore)

                ####

                dd=info[:,1].cpu().numpy().tolist()

                ed=target-fore
                ed=ed.mean(dim=-1)
                ed=ed.cpu().numpy().tolist()
                print(ed)
                e=e+ed
                d=d+dd

                
                

                ####


                #print(fore, target)
                for j in range(target.shape[0]):
                    if target.shape[0]!=7:
                        break
                    #print(input.shape,target.shape[0])
                    w_f[int(info[j][0])].append(fore[j].cpu().numpy().tolist())
                    #print(w_f[int(info[j][0])])
                    w_t[int(info[j][0])].append(target[j].cpu().numpy().tolist())
                    # if int(info[j][0])==0:
                    #     w_f[0].append(fore[j].cpu().numpy().tolist())
            # print(e)
            e=np.asarray(e)
            d=np.asarray(d)
            print(e.shape)
            np.save('error18.npy',e)
            np.save('dinfo18.npy',d)
            print('saved')
            print('w_f=')
            print(w_f)
            print('w_t=')
            print(w_t)

            # np.save('week_days_t.npy',np.asarray(w_t))
            # np.save('week_days_f.npy',np.asarray(w_f))
                
            print('Test loss: '+str(t_loss/i)+'l1: '+str(t_l1/i))
            # print('wday_f=')
            # print(weekday_f)
            # print('wday_t=')
            # print(weekday_t)
            # print('wend_f=')
            # print(weekend_f)
            # print('wend_t=')
            # print(weekend_t)
    def test(self):
        self.model.load_state_dict(torch.load('./checkpoints/simple_trans_15m_withprototype_3atten_5expand_206_rmse304.model'))
        test_set=np.load('./data/test_data.npy',allow_pickle=True)
        test_input=test_set[:,:,4:9].astype(np.float)
        info=test_set[:,0,[9,10]]
        input=torch.from_numpy(test_input).type(torch.float).cuda()
        input[:,:,4]=input[:,:,4]/50
        input=input.permute(0,2,1)
        fore=self.model(input)
        fore=fore.squeeze()
        print(fore)




if __name__=='__main__':

    model=simple()
    dummyinput=torch.FloatTensor(1,5,4)
    print(dummyinput)


