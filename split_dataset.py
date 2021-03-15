import os
import shutil

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir,'data/validation')
gnn_dir =  os.path.join(data_dir,'gnnet_data_set_validation')
new_data_dir = os.path.join(data_dir,'split_dir')
if(not os.path.exists(new_data_dir)):
    os.mkdir(new_data_dir)
# os.mkdir(new_data_dir)
training_data_elems = os.listdir(gnn_dir)

### SPLIT DATA INTO DIFFERENT DIRECTORIES ###

j = 0          # iterator counting the number of samples in each folder
folder_num = 0 # folder for each  
for i, data in enumerate(training_data_elems):
    
    if j == 0:    # make a new folder
        target_path = os.path.join(new_data_dir,str(folder_num))
        os.mkdir(target_path)

    if (data == "graphs") or (data == "routings"):
        continue

    copy_file = os.path.join(gnn_dir,data)
    file_dest = os.path.join(target_path)
    shutil.copy(copy_file,file_dest)

    j+=1
    if (j==10):
        j=0
        folder_num +=1


### MOVE GRAPHS AND ROUTINGS INTO THOSE DIRECTORIES ###
graphs_dir = os.path.join(gnn_dir,"graphs")
routing_dir = os.path.join(gnn_dir,"routings")

# get a list of new directories and copy graphs and routings into those directories
for new_fold in os.listdir(new_data_dir):
    f_dir = os.path.join(new_data_dir,new_fold)
    f_graph_dir = os.path.join(f_dir,"graphs")
    f_route_dir = os.path.join(f_dir,"routings")
    shutil.copytree(graphs_dir,f_graph_dir)
    shutil.copytree(routing_dir,f_route_dir)




