import os
import torch
import shutil
from joblib import delayed,Parallel
from read_dataset import NetworkDataset,get_dataloader


cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir,'data/validation')
split_data_dir = os.path.join(data_dir,'split_dir')
pt_dir = os.path.join(data_dir,"pt_dir")


if not(os.path.exists(os.path.join(data_dir,pt_dir))):
    os.mkdir(pt_dir)


def pt_generator(split_data_dir,pt_dir,i,fdir):

    local_path = os.path.join(split_data_dir,fdir)
    dataset = NetworkDataset(local_path)
    dloader = get_dataloader(dataset,1)

    for j,batch in enumerate(dloader):
        file_str = "folder_" + str(i) + \
                   "sample_" + str(j) + \
                   ".pt"
        save_path = os.path.join(pt_dir,file_str)
        torch.save(batch,save_path)


Parallel(n_jobs=6)(delayed(pt_generator)(split_data_dir,pt_dir,i,f_dir) for i,f_dir in enumerate(os.listdir(split_data_dir)))


# for i,file_dir in enumerate(os.listdir(split_data_dir)):

#     local_path = os.path.join(split_data_dir,file_dir)
#     dataset = NetworkDataset(local_path)
#     dloader = get_dataloader(dataset,1)

#     for j,batch in enumerate(dloader):
#         file_str = "folder_" + str(i) + \
#                    "sample_" + str(j) + \
#                    ".pt"
#         save_path = os.path.join(pt_dir,file_str)
#         if os.path.exists(save_path)
#         torch.save(batch,save_path)