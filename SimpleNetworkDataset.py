from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np


class NetworkDataset(Dataset):
    def __init__(self, path, shuffle=False):
        super(NetworkDataset, self).__init__()
       
        self.file_list = os.listdir(path)
        self.fpath = path
        self.n_sample = len(self.file_list)


    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        
        x,y = torch.load(os.path.join(self.fpath,self.file_list[index]))

        return x,y

def get_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size, shuffle)

