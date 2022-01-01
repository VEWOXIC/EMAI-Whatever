from model.multiscale import multiscale
import torch
import numpy as np
from loader.dataloader import d_set
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

class experiment(object):
    def __init__(self) -> None:
        super().__init__()
        self.lr=0.0001
        self.batch_size=24
        self.epochs=200
        self.model = self._build_model().cuda()
        self._get_data(0.7)

    def _build_model(self):
        model=multiscale()
        return model
    
    def _get_data(self,tt_ratio):
        input=np.load('./data/inputno_nan.npy',allow_pickle=True)
        output=np.load('./data/outputno_nan.npy',allow_pickle=True)
        l=input.shape[0]
        train_input=input[:int(l*tt_ratio)]
        train_output=output[:int(l*tt_ratio)]
        test_input=input[int(l*tt_ratio):]
        test_output=output[int(l*tt_ratio):]
        self.train_set=d_set(train_input,train_output)
        self.test_set=d_set(test_input,test_output)
        self.train_loader=DataLoader(self.train_set,batch_size=self.batch_size,shuffle=True)
        self.test_loader=DataLoader(self.test_set,batch_size=self.batch_size,shuffle=False)

        return self.train_loader,self.test_loader

    def _get_optim(self):
        return torch.optim.Adam(params=self.model.parameters(),lr=self.lr)

    
    def train(self):
        my_optim=self._get_optim()
        bestloss=1000000
        
        lossf=nn.L1Loss().cuda()

        for epoch in range(self.epochs):
            t_loss=0
            for i,(input,target) in enumerate(self.train_loader):
                input=input.cuda()
                co=input[:,0:4]
                woy=F.one_hot(input[:,4,3].to(torch.int64),num_classes=52)
                dow=F.one_hot(input[:,4,1].to(torch.int64),num_classes=7)
                hod=F.one_hot(input[:,4,0].to(torch.int64),num_classes=24)
                target=target.cuda()
                self.model.zero_grad()

                fore=self.model(hod.unsqueeze(dim=-1).to(torch.float32),dow.unsqueeze(dim=-1).to(torch.float32),woy.unsqueeze(dim=-1).to(torch.float32),co)
                fore=fore.squeeze()
                target=target.squeeze()
                #print(fore,target)
                loss=lossf(fore,target)

                loss.backward()
                my_optim.step()

                t_loss+=loss
                #print(loss)

            print('Epoch:'+str(epoch)+' loss: '+str(t_loss/i))
            if t_loss/i<bestloss:
                bestloss=t_loss/i
                torch.save(self.model.state_dict(),'./checkpoints/multi2.model')

            with torch.no_grad():
                t_loss=0
                for i,(input,target) in enumerate(self.test_loader):
                    input=input.cuda()
                    co=input[:,0:4]
                    woy=F.one_hot(input[:,4,3].to(torch.int64),num_classes=52)
                    dow=F.one_hot(input[:,4,1].to(torch.int64),num_classes=7)
                    hod=F.one_hot(input[:,4,0].to(torch.int64),num_classes=24)
                    target=target.cuda()
                    fore=self.model(hod.unsqueeze(dim=-1).to(torch.float32),dow.unsqueeze(dim=-1).to(torch.float32),woy.unsqueeze(dim=-1).to(torch.float32),co)
                    fore=fore.squeeze()
                    target=target.squeeze()
                    loss=lossf(fore,target)
                    t_loss+=loss
                print('Test loss: '+str(t_loss/i))

    def test(self):
        self.model.load_state_dict(torch.load('./checkpoints/multi2.model'))

        # for name,param in self.model.hourofday_block.embedding.named_parameters():
        #     if param.requires_grad:

        #         plt.plot(param.data.cpu().detach().squeeze().numpy())
        #         plt.savefig('./weights_hod.jpg')
        # for name,param in self.model.dayofweek_block.embedding.named_parameters():
        #     if param.requires_grad:

        #         plt.plot(param.data.cpu().detach().squeeze().numpy())
        #         plt.savefig('./weights_dow.jpg')
        for name,param in self.model.weekofyear_block.embedding.named_parameters():
            if param.requires_grad:

                plt.plot(param.data.cpu().detach().squeeze().numpy())
                plt.savefig('./weights_woy.jpg')
        lossf=nn.L1Loss().cuda()
        with torch.no_grad():
            t_loss=0
            for i,(input,target) in enumerate(self.test_loader):
                input=input.cuda()
                target=target.cuda()
                fore=self.model(input)
                fore=fore.squeeze()
                target=target.squeeze()
                loss=lossf(fore,target)
                t_loss+=loss
                #print(fore, target)
            print('Test loss: '+str(t_loss/i))



            



if __name__=='__main__':

    model=simple()
    dummyinput=torch.FloatTensor(1,5,4)
    print(dummyinput)


