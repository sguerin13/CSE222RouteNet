from SimpleNetworkDataset import NetworkDataset
import torch
import torch.nn as nn
import tqdm
import os

train_path = "/home/steve/Documents/GitHub/CSE222a/CSE222RouteNet/data/training/pt_dir"
train_set = NetworkDataset(train_path)
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                batch_size=1,
                                                shuffle=True)

total_max_cap = 0
total_max_bw  = 0
total_max_del = 0
total_max_jit = 0

for i,batch in enumerate(train_loader):
    if i % 1000 == 0:
        if (i != 0):
            max_dict = {'cap':total_max_cap,
                        'bw': total_max_bw,
                        'del': total_max_del,
                        'jit':total_max_jit}
            torch.save(max_dict,os.path.join(os.getcwd(),'max_values.pt'))

        print(i)
    x,y = batch
    max_cap = max(x['link_capacity'])
    max_bw = max(x['bandwith'])
    max_del = max(y[0])
    max_jit = max(y[1])

    if max_cap > total_max_cap:
        total_max_cap = max_cap

    if max_bw > total_max_bw:
        total_max_bw = max_bw

    if max_del > total_max_del:
        total_max_del = max_del

    if max_jit > total_max_jit:
        total_max_jit = max_jit


max_dict = {'cap':total_max_cap,
            'bw': total_max_bw,
            'del': total_max_del,
            'jit':total_max_jit}

torch.save(max_dict,os.path.join(os.getcwd(),'max_values.pt'))


## testing routine
x = torch.load(os.path.join(os.getcwd(),'max_values.pt'))