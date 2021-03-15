import torch
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from CustomLoss import CustomLoss,MAPE,TwoTerm,MAPE2T
from weight_init import weight_init
import os
import time

class Runner():
    '''
    Helper class that makes it a bit easier and cleaner to define the training routine
    
    '''
    
    def __init__(self,model,train_set,test_set,opts,max_vals,log_name):
        self.model = model  # neural net
        


        self.log_name = log_name
        log_path = os.getcwd() + '/logs/' + self.log_name + '.txt'
        model_path = os.getcwd() + '/saved_models/'
        self.log = Logger(log_path,model_path)
        
        # device agnostic code snippet
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.float()
        

        self.max_val_dict = max_vals
        self.epochs = opts['epochs']
        self.optimizer = torch.optim.SGD(model.parameters(), opts['lr'],momentum=.9) # optimizer method for gradient descent
        self.tr_criterion = TwoTerm()                                  # loss function
        self.test_criterion = MAPE2T()
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                        batch_size=opts['batch_size'],
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                       batch_size=opts['batch_size'],
                                                       shuffle=True)
        
    def train(self):
        self.model.train() #put model in training mode
        for epoch in range(self.epochs):
            self.tr_loss = []
            self.model.train()
            st_time = time.time()
            for i, (data,labels) in tqdm.tqdm_notebook(enumerate(self.train_loader),
                                                   total = len(self.train_loader)):
                if (i % 1000 == 0):
                    print(i)
                    end_time = time.time()
                    print(end_time - st_time)
                    st_time = end_time


                # data, labels = data.to(self.device),labels.to(self.device)
                labels[0] = torch.unsqueeze(torch.tensor(labels[0]),1)/(self.max_val_dict['del']) # normalize
                labels[0] = labels[0].to(self.device)
                labels[0] = labels[0].float()

                labels[1] = torch.unsqueeze(torch.tensor(labels[1]),1)/(self.max_val_dict['jit'])
                labels[1] = labels[1].to(self.device)
                labels[1] = labels[1].float()

                data['links'] = torch.unsqueeze(torch.tensor(data['links']),1).to(self.device)
                data['paths'] = torch.unsqueeze(torch.tensor(data['paths']),1).to(self.device)                 
                data['sequences'] = torch.unsqueeze(torch.tensor(data['sequences']),1).to(self.device)
                data['link_capacity'] = torch.unsqueeze(torch.tensor(data['link_capacity']).float(),axis=1)/(self.max_val_dict['cap'].float())
                data['link_capacity'] = data['link_capacity'].to(self.device)
                data['link_capacity'] = data['link_capacity'].float()
                data['bandwith']= torch.unsqueeze(torch.tensor(data['bandwith']).float(), axis=1)/(self.max_val_dict['bw'].float())
                data['bandwith'] = data['bandwith'].to(self.device)
                data['bandwith'] = data['bandwith'].float()
                
                data['n_links'] = data['n_links'][0].to(self.device)
                data['n_paths'] = data['n_paths'][0].to(self.device)
                data['packets'] = data['packets'][0].to(self.device)

                self.optimizer.zero_grad()  
                d,j = self.model(data)   
                loss = self.tr_criterion(d,j,labels) 
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),2) # clip the gradient                       
                self.optimizer.step()                  
                self.tr_loss.append(loss.item())       
            
            self.test(epoch) # run through the validation set
            self.log.save_model(self.model,self.log_name,epoch)
        
    def test(self,epoch):
            
            self.model.eval()    # puts model in eval mode - not necessary for this demo but good to know
            self.test_delay = []
            self.test_jitter = []
            for i, (data, labels) in enumerate(self.test_loader):
                
                if (i % 1000 == 0):
                    print(i)

                # data, labels = data.to(self.device),labels.to(self.device)
                                # data, labels = data.to(self.device),labels.to(self.device)
                labels[0] = torch.unsqueeze(torch.tensor(labels[0]),1).to(self.device)
                labels[1] = torch.unsqueeze(torch.tensor(labels[1]),1).to(self.device)

                data['links'] = torch.unsqueeze(torch.tensor(data['links']),1).to(self.device)
                data['paths'] = torch.unsqueeze(torch.tensor(data['paths']),1).to(self.device)                 
                data['sequences'] = torch.unsqueeze(torch.tensor(data['sequences']),1).to(self.device)
                data['link_capacity'] = torch.unsqueeze(torch.tensor(data['link_capacity']).float(),axis=1).to(self.device)
                data['bandwith']= torch.unsqueeze(torch.tensor(data['bandwith']).float(), axis=1).to(self.device)
                data['n_links'] = data['n_links'][0].to(self.device)
                data['n_paths'] = data['n_paths'][0].to(self.device)
                data['packets'] = data['packets'][0].to(self.device)
                
                # pass data through network
                # turn off gradient calculation to speed up calcs and reduce memory
                with torch.no_grad():
                    d,j = self.model(data)
                
                loss = self.test_criterion(d,j, labels)
                self.test_delay.append(loss[0].detach().cpu().numpy())
                self.test_jitter.append(loss[1].detach().cpu().numpy())

            log_str = 'epoch: ' + str(epoch + 1) + ', train loss: ' + str(np.mean(self.tr_loss))\
                      + ', val delay percent error: ' + str(np.mean(self.test_delay)) \
                      + ', val jitter percent error: ' + str(np.mean(self.test_jitter))
            self.log.update_log(log_str)
            print('epoch: {}, train loss: {}, val delay percent error: {}, val jitter percent error: {}'.format( 
                  epoch+1, np.mean(self.tr_loss), np.mean(self.test_delay), np.mean(self.test_jitter)))


class Logger():
    def __init__(self,log_path,model_save_path):
        self.log_path = log_path
        self.model_save_path = model_save_path
    
    def update_log(self,log_str):
        with open(self.log_path,'a+') as f:
            f.write("\n")
            f.write(log_str)

    def gen_save_str(self,aux,epoch):
        str_path = self.model_save_path + aux + '_epoch_' + str(epoch) + '.pt'
        return str_path

    def save_model(self,model,aux,epoch):

        save_str = self.gen_save_str(aux,epoch)
        torch.save(model,save_str)
        
    
            

