import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json

class ShapeNetDataset(data.Dataset):

    def __init__(self,
                 root,
                 npoints = 500,
                 ):
        self.npoints = npoints
        self.root = root
        cats = os.listdir(self.root)
        
        self.point_dict = {}
        i = 0
        for j,cat in enumerate(cats):
            pt_files = os.listdir(self.root+'\\'+cat)
            for pts in pt_files:
                self.point_dict[i] = {}
                self.point_dict[i]['cat'] = j
                self.point_dict[i]['points'] = \
                     np.loadtxt(self.root+'\\'+cat+'\\'+pts).astype(np.float32)
                pt_len = self.point_dict[i]['points'].shape[0]
                # shuffle the point order
                np.random.shuffle(self.point_dict[i]['points'])
                self.point_dict[i]['sample_pts'] = self.point_dict[i]['points'][:self.npoints,:]
                i+=1

    def __getitem__(self, index):
        pts = self.point_dict[index]['sample_pts']

        label = self.point_dict[index]['cat']
        pts = pts - np.expand_dims(np.mean(pts, axis = 0), 0) # center

        dist = np.max(np.sqrt(np.sum(pts ** 2, axis = 1)),0)
        pts = pts / dist #scale

        # convert to tensor
        pts = torch.from_numpy(pts)
        label = torch.from_numpy(np.array(label).astype('float32'))
        label = label.long()
        return pts, label

    def __len__(self):
        return len(self.point_dict.keys())