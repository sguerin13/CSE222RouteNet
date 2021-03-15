import torch.nn as nn
import torch

class CustomLoss(nn.Module):
    def __init__(self): 
        super(CustomLoss, self).__init__()
        
    def forward(self, y_pred, y, num_packet):
        # old inputs: self, delay, delay_t, jitter, jitter_t, num_packet
        
        # I assume the input 5 parameters are all vectors of 1 by 200 for example,
        # representing the training data and true value of delay and jitter, also the number of packets received
        # calculate the negative log-likelihood and return their average
        delay = torch.unsqueeze(y_pred[:,0],1)
        jitter = torch.unsqueeze(y_pred[:,1],1)

        c = torch.log(torch.expm1(torch.tensor(0.098)))
        sig = torch.add(torch.nn.functional.softplus(torch.add(c,jitter)),torch.tensor(1e-9))
        jitter = torch.pow(sig,2)
        
        # delay_t = torch.unsqueeze(torch.tensor(y[0]),1)
        # jitter_t = torch.unsqueeze(torch.tensor(y[1]),1)
        delay_t = y[0]
        jitter_t = y[1]
        
        num_packet = torch.unsqueeze(torch.tensor(num_packet),1)
        
        # nll = num_packet * ((jitter_t + (delay_t - delay)**2)/(2*jitter**2) + torch.log(jitter))

        nll =               torch.add(
                                    torch.mul(num_packet,
                                              torch.div(torch.add(jitter_t,torch.pow(torch.sub(delay_t,delay),2)),
                                                        torch.mul(torch.pow(jitter,2),2))
                                             ),
                                    torch.log(jitter)
                                    )

        out = torch.div(torch.sum(nll),torch.tensor(1e6))

        return out


class MAPE(nn.Module):
    def __init__(self):
        super(MAPE,self).__init__()

    def forward(self,y_pred,y):
        
        delay = torch.unsqueeze(y_pred[:,0],1)
        jitter = torch.unsqueeze(y_pred[:,1],1)

        c = torch.log(torch.expm1(torch.tensor(0.098)))
        sig = torch.add(torch.nn.functional.softplus(torch.add(c,jitter)),torch.tensor(1e-9))
        jitter = torch.pow(sig,2)
        
        # delay_t = torch.unsqueeze(torch.tensor(y[0]),1)
        # jitter_t = torch.unsqueeze(torch.tensor(y[1]),1)
        delay_t = y[0]
        jitter_t = y[1]

        # add the .0000001 to the end to prevent div by 0
        delay_mape  = torch.mean(torch.div(torch.abs(torch.sub(delay_t,delay)),torch.add(delay_t,.000000001)))
        jitter_mape = torch.mean(torch.div(torch.abs(torch.sub(jitter_t,jitter)),torch.add(jitter_t,.000000001)))

        return delay_mape,jitter_mape


class MAPE2T(nn.Module):
    def __init__(self):
        super(MAPE2T,self).__init__()

    def forward(self,y_d,y_j,y):
        
        delay = y_d
        jitter = y_j

        # delay_t = torch.unsqueeze(torch.tensor(y[0]),1)
        # jitter_t = torch.unsqueeze(torch.tensor(y[1]),1)
        delay_t = y[0]
        jitter_t = y[1]
        # print(torch.mean(jitter),torch.mean(jitter_t))
        # add the .0000001 to the end to prevent div by 0
        delay_mape  = torch.mean(torch.div(torch.abs(torch.sub(delay_t,delay)),delay_t))
        jitter_mape = torch.mean(torch.div(torch.abs(torch.sub(jitter_t,jitter)),jitter_t))

        return delay_mape,jitter_mape


class TwoTerm(nn.Module):
    def __init__(self):
        super(TwoTerm,self).__init__()

    def forward(self,y_d,y_j,y):
        
        delay = y_d
        jitter = y_j

        # delay_t = torch.unsqueeze(torch.tensor(y[0]),1)
        # jitter_t = torch.unsqueeze(torch.tensor(y[1]),1)
        delay_t = y[0]
        jitter_t = y[1]

        # add the .0000001 to the end to prevent div by 0
        delay_mse = torch.mean(torch.pow(torch.sub(delay_t,delay),2))
        jitter_mse = torch.mean(torch.pow(torch.sub(jitter_t,jitter),2))
        e_total = torch.add(torch.div(delay_mse,2),torch.div(jitter_mse,2))

        return e_total
        
