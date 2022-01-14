from model.simple_trans import simple
import torch
import numpy as np
from loader.dataloader import day_set
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from loader.data_preprocessing import *
from model.imputation import SCI_Point_Mask
import argparse 
def parse_args():
    parser = argparse.ArgumentParser(description='emai')
    parser.add_argument('--model-name', type=str, default='head6_hid48_drop0.6', help='model name')
    parser.add_argument('--layer', type=int, default=3, help='layer of self attention block') 
    parser.add_argument('--hidden', type=int, default=48, help='hidden size of variates')
    parser.add_argument('--head', type=int, default=6, help='head size of variates')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout of hidden layer')
    parser.add_argument('--lr', type=float, default=6e-3, help='learning rate')
    parser.add_argument('--batch', type=int, default=7, help='batch size')
    parser.add_argument('--proto', type=int, default=50, help='scale of prototype')
    parser.add_argument('--seed', type=int, default=99, help='random seed')
    parser.add_argument('--ensemble', action='store_false', help='ensemble top 5 models')
    parser.add_argument('--kernel', default=3, type=int, help='kernel size')
    parser.add_argument('--dilation', default=1, type=int, help='dilation')
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--hidden-size', default=1, type=int, help='hidden channel of module')
    # parser.add_argument('--dropout', type=float, default=0.8, help='dropout of hidden layer')

    args = parser.parse_args()
    return args

args = parse_args()

class experiment(object):
    def __init__(self,training_csv):
        super().__init__()
        self.lr=args.lr
        self.batch_size=args.batch 
        self.epochs=1000
        self.model, self.SCI_Imputation = self._build_model()
        self.model = self.model.cuda()
        self.SCI_Imputation = self.SCI_Imputation.cuda()
        self.ensemble = args.ensemble ##todo:modify here
        print('ensemble:',self.ensemble)
        self.training_file=training_csv
        input, output, self.prototype = train_input_output(self.training_file)
    def _build_model(self):
        model = simple(args, args.layer, 5)
        part = [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]  # Best model
        sci_imputation = SCI_Point_Mask(args, num_classes=96, input_len=96, input_dim=1,
                                        number_levels=len(part),
                                        number_level_part=part, num_layers=3, concat_len=None).cuda()

        return model, sci_imputation

    def get_index(self,years,dayofweek,dates):
            day_list = [[] for i in range(7)]
            for day in [0,1,2,3,4,5,6]:
                for i,(year,dow,date) in enumerate(zip(years,dayofweek,dates)):
                    holiday = (int(year[0][:4]) == 2020 and int(date[0]) in [1,25,27,28,95,101,102,104,121,122,177,183,275,276,360,361]) or  (int(year[0][:4]) ==2021 and int(date[0]) in [1,44,46,92,93,95,96,121,122,139,274,287,359,361])
                    if day in [0,1,2,3,4,5]:
                        if int(dow[0]) == day and not holiday:
                            day_list[day].append(i)
                    else:
                        if int(dow[0]) == 6 or holiday:
                            day_list[day].append(i)
            return day_list
    def data_imputation(self, data):

        temperature = [0]
        Humidity = [1]
        UV_Index = [2]
        Average_Rainfall = [3]
        Imputed_Data = []
        for covariates in data:
            input_tmp = np.array(covariates)

            T = covariates[:, temperature]  # 96, 4
            H = covariates[:, Humidity]
            U = covariates[:, UV_Index]
            A = covariates[:, Average_Rainfall]


            if np.where(T==-1)[0].shape[0]>0:#np.isnan(T).any():
                # loc = np.argwhere(np.isnan(T[..., 0]))  # 96,
                # loc = np.where(T == -1)
                # loc = list(loc)
                # loc = [int(_) for _ in loc]
                loc = np.where(T == -1)
                loc = list(loc[0])
                loc = [int(_) for _ in loc]
                self.SCI_Imputation.load_state_dict(torch.load('./checkpoints/tem_impute_S_L1.model'))
                T[loc] = 0.0
                T_i = torch.from_numpy(T)
                T_i = T_i.unsqueeze(0)  # 1,96,4
                T_i = T_i.float().cuda()
                fore = self.SCI_Imputation(T_i)  # 1,96,1
                fore = fore.squeeze()  # 96
                fore = fore.detach().cpu().numpy()
                T[loc, 0] = fore[loc]  # 96,4

            if np.where(H==-1)[0].shape[0]>0:#np.isnan(H).any():
                # loc = np.argwhere(np.isnan(H[..., 0]))  # 96,
                # loc = list(loc)
                # loc = [int(_) for _ in loc]
                loc = np.where(H == -1)
                loc = list(loc[0])
                loc = [int(_) for _ in loc]
                self.SCI_Imputation.load_state_dict(torch.load('./checkpoints/hum_impute_S_L1.model'))
                H[loc] = 0.0
                H_i = torch.from_numpy(H)
                H_i = H_i.unsqueeze(0)  # 1,96,4
                H_i = H_i.float().cuda()
                fore = self.SCI_Imputation(H_i)  # 1,96,1
                fore = fore.squeeze()  # 96
                fore = fore.detach().cpu().numpy()
                H[loc, 0] = fore[loc]

            if np.where(U==-1)[0].shape[0]>0:#np.isnan(U).any():
                # loc = np.argwhere(np.isnan(U[..., 0]))  # 96,
                # loc = list(loc)
                # loc = [int(_) for _ in loc]
                loc = np.where(U == -1)
                loc = list(loc[0])
                loc = [int(_) for _ in loc]
                self.SCI_Imputation.load_state_dict(torch.load('./checkpoints/uv_impute_S_L1.model'))
                U[loc] = 0.0
                U_i = torch.from_numpy(U)
                U_i = U_i.unsqueeze(0)  # 1,96,4
                U_i = U_i.float().cuda()
                fore = self.SCI_Imputation(U_i)  # 1,96,1
                fore = fore.squeeze()  # 96
                fore = fore.detach().cpu().numpy()
                U[loc, 0] = fore[loc]

            if np.where(A==-1)[0].shape[0]:#np.isnan(A).any():
                # loc = np.argwhere(np.isnan(A[..., 0]))  # 96,
                # loc = list(loc)
                # loc = [int(_) for _ in loc]
                loc = np.where(A == -1)
                loc = list(loc[0])
                loc = [int(_) for _ in loc]
                self.SCI_Imputation.load_state_dict(torch.load('./checkpoints/af_impute_S_L1.model'))
                A[loc] = 0.0
                A_i = torch.from_numpy(A)
                A_i = A_i.unsqueeze(0)  # 1,96,4
                A_i = A_i.float().cuda()
                fore = self.SCI_Imputation(A_i)  # 1,96,1
                fore = fore.squeeze()  # 96
                fore = fore.detach().cpu().numpy()
                A[loc, 0] = fore[loc]

            T = T[..., 0]  # 96
            H = H[..., 0]  # 96
            U = U[..., 0]  # 96
            A = A[..., 0]  # 96

            imputed = np.vstack((T, H, U, A)).T
            imputed = imputed[np.newaxis, :]
            Imputed_Data.append(imputed)

        Imputed_Data = np.vstack(Imputed_Data)
        cat_data =  data[..., 4]
        cat_data = cat_data[:,:,np.newaxis]
        Imputed_Data = np.concatenate((Imputed_Data, cat_data), axis=2)

        return Imputed_Data

    def _get_data(self,tt_ratio,training_file=None):
        # data 1: no outliner, no wrong holiday 
        # input=np.load('data/input_no_outliner_fix.npy',allow_pickle=True)
        # # input=np.load('data/input_no_outliner_fix_allmean.npy',allow_pickle=True)
        # output=np.load('data/output_no_outliner.npy',allow_pickle=True)
        input, output, self.prototype = train_input_output(training_file)

        # data 2: has outliner, no wrong holiday
        # input=np.load('data/input_fix.npy',allow_pickle=True)
        # output=np.load('data/output_fix.npy',allow_pickle=True)

        # data 3: has outliner, wrong holiday 
        # input=np.load('data/input_18month_imputed_18prototype.npy',allow_pickle=True)
        # output=np.load('./data/output_18month_imputed.npy',allow_pickle=True)

        # input=np.load('./data/input_18month_imputed_18prototype.npy',allow_pickle=True)
        # input=np.load('./data/input_fix.npy',allow_pickle=True)

        # input=np.load('./data/input_no_outliner_fix_allmean.npy',allow_pickle=True)
        day_list = self.get_index(input[:,:,0],input[:,:,9],input[:,:,10])
        input = input[day_list[self.day]]
        input=input[:,:,[4,5,6,7,8,9,10]].astype(np.float)
        input[:,:,4]=input[:,:,4]/args.proto

        # output=np.load('./data/output_18month_imputed.npy',allow_pickle=True)
        # output=np.load('./data/output_no_outliner.npy',allow_pickle=True)
        # output=np.load('./data/output_18month_imputed_18prototype.npy',allow_pickle=True)
        output = output[day_list[self.day]]

        self.bias=output.min()
        self.std=output.std()
        self.out_scale = 1

        train_input,test_input,train_output,test_output=train_test_split(input,output,test_size=tt_ratio,random_state=args.seed)
        l=input.shape[0]
        # print(train_input[:2,10,1],test_input[2,2],test_output[1,1],train_input.shape,test_input.shape,train_output.shape,test_output.shape) #(44, 96, 7) (5, 96, 7) (44, 24) (5, 24)
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
            self._get_data(0.1,self.training_file)
            self.train_a_day()
            self.validate_a_day()

    def train_a_day(self):
        my_optim=self._get_optim()
        bestloss=1000000
        lossf=nn.MSELoss().cuda()
        l1=nn.L1Loss().cuda()
        for epoch in range(self.epochs):
            self.model.train()
            t_loss=0
            if epoch%100==0 & epoch!=0:
                my_optim.lr=my_optim.lr/2 #jåŠ å…¥åŠ¨æ€è°ƒæ•´ l1loss 514

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
                    torch.save(self.model.state_dict(),'./checkpoints/day{}.{}'.format(self.day,args.model_name))

    def validate_a_day(self):
        self.model.load_state_dict(torch.load('./checkpoints/day{}.{}'.format(self.day,args.model_name)))
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
            # print('kkkk',np.mean(out_mse),np.std(out_mse),i+1,len(out_mse))
            print('Day:%d  '%self.day,'batch:',i+1, 'Test mse: '+str(t_loss.item()*self.out_scale/(i+1))+' mae: '+str(t_l1.item()*self.out_scale/(i+1)))

        return i+1, t_loss, t_l1

    def ensemble_predict(self,input, model_names):
        # input [b,c,t]
        output_list = torch.zeros([len(model_names), input.size(0), 24])
        for i,name in enumerate(model_names):
            model = simple(args, args.layer,5) 
            model.load_state_dict(torch.load('%s'%name))
            model = model.cuda()
            model.eval()
            fore = model(input)
            fore = fore.squeeze()
            output_list[i] = fore
        return output_list.mean(0)
        # return torch.from_numpy(np.array(output_list)).cuda()

    def validate_a_day_ensemble(self,name_of_models):
        # Validate one head
        lossf=nn.MSELoss().cuda()
        l1=nn.L1Loss().cuda()
        with torch.no_grad():
            t_loss=0
            t_l1=0
            out_mse = []
            for i,(input,target) in enumerate(self.test_loader):
                raw_input=input.cuda()
                input=raw_input[:,:,:5]
                info=raw_input[:,0,[5,6]]
                target=target.cuda()
                input=input.permute(0,2,1)
                fore=self.ensemble_predict(input,name_of_models)
                fore=fore.cuda().squeeze()
                loss = torch.mean(loss)
                t_loss+=loss
                out_mse.append(loss.cpu())
                t_l1+=l1(fore,target)
            print('Day:%d  '%self.day,'batch:',i+1, 'Test mse: '+str(t_loss.item()*self.out_scale/(i+1))+' mae: '+str(t_l1.item()*self.out_scale/(i+1)))

        return i+1, t_loss, t_l1

    def validate(self):
        models = [simple(args, args.layer, 5) for i in range(7)]
        batch_total, mse_total, l1_total = 0,0,0
        for i in range(7):
            self.day = i
            self._get_data(0.1,self.training_file)
            if self.ensemble:
                name_of_models = ['./checkpoints/day{}.{}_s{}'.format(self.day,args.model_name, i) for i in range(1,5)]
                batch, mse, l1 = self.validate_a_day_ensemble(name_of_models)
                batch_total+=batch
                mse_total+=mse
                l1_total+=l1
            else:
                models[i].load_state_dict(torch.load('./checkpoints/day{}.{}'.format(i,args.model_name)))
                models[i].cuda()
                self.model = models[i]
                batch, mse, l1 = self.validate_a_day()

        print('Batch:',batch_total, 'Test mse: '+str(mse_total.item()*self.out_scale/batch_total)+' mae: '+str(l1_total.item()*self.out_scale/batch_total))

    def test(self):
        with torch.no_grad():
            # Only support ensemble
            # test_set=np.load('./data/test_data.npy',allow_pickle=True)
            test_set = test_data_preprocessing(self.prototype)
            day_list = self.get_index(test_set[:,:,0],test_set[:,:,9],test_set[:,:,10])
            test_input=test_set[:,:,4:9].astype(np.float)
            info=test_set[:,0,[9,10]]
                    # Data imputation -------------------------------------------------
            test_input = self.data_imputation(test_input)
            # -------------------------------------------------
            input=torch.from_numpy(test_input).type(torch.float).cuda()
            input[:,:,4]=input[:,:,4]/args.proto
            input=input.permute(0,2,1)

            fore_output = torch.zeros([input.size(0),24])
            fore_output = fore_output.squeeze()

            for day in range(7):
                if len(day_list[day]) != 0:
                    name_of_models = ['./final_checkpoints/day{}.{}_s{}'.format(day,args.model_name, i) for i in range(1,6)]
                    fore = self.ensemble_predict(input[day_list[day]],name_of_models)
                    fore=fore.squeeze()
                    fore_output[day_list[day]] = fore
            df=pd.DataFrame({'Timestamp':test_set[:,::4,3].flatten(),'CoolingLoad':fore_output.flatten().cpu().detach().numpy()})
            df.to_csv('./A-P10005_output.csv',index=False,sep=',')
            
            output=np.load('data/output_no_outliner_fix.npy',allow_pickle=True)
            target=torch.tensor(output[-7:]).cuda()
            print(fore_output,target)
            lossf=nn.MSELoss().cuda()
            loss=torch.sqrt(lossf(fore_output.cuda(),target))

            print(loss)

            #print(fore_output)
            # print(fore_output)




