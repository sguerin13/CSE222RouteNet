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
        
        delay_t = torch.unsqueeze(torch.tensor(y[0]),1)
        jitter_t = torch.unsqueeze(torch.tensor(y[1]),1)
        
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