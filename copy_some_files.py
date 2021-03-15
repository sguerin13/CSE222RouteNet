import os
import shutil

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir,'data/validation')
pt_dir =  os.path.join(data_dir,'pt_dir')

new_data_dir = os.path.join(cur_dir,'data/smaller_dataset/validation')
if(not os.path.exists(new_data_dir)):
    os.mkdir(new_data_dir)
# os.mkdir(new_data_dir)
training_data_elems = os.listdir(pt_dir)

### SPLIT DATA INTO DIFFERENT DIRECTORIES ###
stop_num = 10000
for i, data in enumerate(training_data_elems):

    copy_file = os.path.join(pt_dir,data)
    file_dest =new_data_dir
    shutil.copy(copy_file,file_dest)

    if i == stop_num:
        break