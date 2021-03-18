import torch
import torch
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from CustomLoss import CustomLoss,MAPE,TwoTerm,MAPE2T
from weight_init import weight_init
import os
import pickle
import time
import sys


class Evaluator():

    def __init__(self,model,eval_set,max_vals):
        self.model = model  # neural net
        self.eval_set = eval_set
        
        # device agnostic code snippet
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.float()
        
        self.max_val_dict = max_vals
        self.test_loader = torch.utils.data.DataLoader(dataset=self.eval_set,
                                                       batch_size=1,
                                                       shuffle=True)
        # store for later
        self.eval_dict = {'delay_percent':[],
                          'delay_pred': [],
                          'delay_gt':[],
                          'delay_0_abs_error':[],
                          'delay_0_gt':[],
                          'jitter_percent':[],
                          'jitter_pred': [],
                          'jitter_gt':[],
                          'jitter_0_abs_error':[],
                          'jitter_0_gt':[],}
        

    def eval(self,fpath):
            
            self.model.eval()    # puts model in eval mode - not necessary for this demo but good to know
            for i, (data, labels) in enumerate(self.test_loader):
                
                if (i % 1000 == 0):
                    print(i)
                
                if i > 2000:
                    break

                # data, labels = data.to(self.device),labels.to(self.device)
                                # data, labels = data.to(self.device),labels.to(self.device)
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
                
                # pass data through network
                # turn off gradient calculation to speed up calcs and reduce memory
                with torch.no_grad():
                    d,j = self.model(data)
                
                self.percent_error(d,j,labels[0],labels[1])

            # save the percent error dictionary for analysis later
            self.save_eval_dict(fpath)



    def percent_error(self,d,j,d_t,j_t):
        delay = d.flatten()
        jitter = j.flatten()



        delay_t = d_t.flatten()
        jitter_t = j_t.flatten()


        for i in range(len(delay)):

            if delay_t[i] == 0:
                mae_delay = torch.abs(torch.sub(delay[i],delay_t[i]))

                self.eval_dict['delay_0_abs_error'].extend([mae_delay.detach().cpu().numpy()])
                self.eval_dict['delay_0_gt'].extend([delay_t[i].detach().cpu().numpy()])
                self.eval_dict['delay_pred'].extend([delay[i].detach().cpu().numpy()])
            
            else:
                mape_delay = torch.abs(torch.div(torch.sub(delay[i],delay_t[i]),delay_t[i]))

                self.eval_dict['delay_percent'].extend([mape_delay.detach().cpu().numpy()])
                self.eval_dict['delay_gt'].extend([delay_t[i].detach().cpu().numpy()])
                self.eval_dict['delay_pred'].extend([delay[i].detach().cpu().numpy()])

            if jitter_t[i] == 0:
                mae_jitter = torch.abs(torch.sub(jitter[i],jitter_t[i]))
                self.eval_dict['jitter_0_abs_error'].extend([mae_jitter.detach().cpu().numpy()])
                self.eval_dict['jitter_0_gt'].extend([jitter_t[i].detach().cpu().numpy()])
                self.eval_dict['jitter_pred'].extend([jitter[i].detach().cpu().numpy()])
            
            else:
                mape_jitter = torch.abs(torch.div(torch.sub(jitter[i],jitter_t[i]),jitter_t[i]))

                self.eval_dict['jitter_percent'].extend([mape_jitter.detach().cpu().numpy()])
                self.eval_dict['jitter_gt'].extend([jitter_t[i].detach().cpu().numpy()])
                self.eval_dict['jitter_pred'].extend([jitter[i].detach().cpu().numpy()])

    
    def save_eval_dict(self,fpath):
        # open a file, where you ant to store the data
        with open(fpath, 'wb') as file:

            # dump information to that file
            pickle.dump(self.eval_dict, file)


    
            

