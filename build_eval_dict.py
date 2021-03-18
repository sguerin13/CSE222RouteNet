from Performance_Eval import Evaluator
from model import RouteNet
from SimpleNetworkDataset import NetworkDataset
import torch
import torch.nn as nn
import os

eval_path= os.path.join(os.getcwd(),"data/smaller_dataset/validation")
eval_set = NetworkDataset(eval_path)

file_path = os.path.join(os.getcwd(),"saved_models/run_2_epoch_14.pt")
model = torch.load(file_path)

max_vals = torch.load(os.path.join(os.getcwd(),"max_values.pt"))

eval_file = os.path.join(os.getcwd(),"evaluation_data/run_2_e_14.pkl")

evaluator = Evaluator(model,eval_set,max_vals)
evaluator.eval(eval_file)
