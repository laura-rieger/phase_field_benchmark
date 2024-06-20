# load everything
import sys
from segformer_pytorch import Segformer
import configparser
import pickle as pkl
from argparse import ArgumentParser
from copy import deepcopy
from os.path import join as oj
from unet_model import UNet
import torch
import os

# from segformer_pytorch import Segformer
import torch.utils.data
from data_loading import H5pyDataset
import pandas as pd
# from torchvision import transforms
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import optim
from torch.utils.data import DataLoader
from torch import nn
import data_utils
import platform
print("What")
# set up device
cuda = torch.cuda.is_available()
torch.set_num_threads(1)
device = torch.device("cuda" if cuda else "cpu")

# set up config parser
config = configparser.ConfigParser()
config.read('../config.ini')

# set up model path
model_path = config["PATHS"]["model_unet_path"]

data_folder = config["DATASET"]["dataPath"]
import numpy as np



# load all pkl files
fnames = sorted([oj(model_path, fname) for fname in os.listdir(model_path) if "pkl" in fname]) 
results_list = [pd.Series(pkl.load(open(fname, "rb"))) for fname in (fnames)] 
results = pd.concat(results_list, axis=1).T.infer_objects()
results = results[results.exp_name == 'PhaseFieldPrediction']
results.reset_index(inplace=True)
# allocate nm
test_dataset = H5pyDataset( oj(data_folder, "phase_field_test.h5"), offset=10, start_offset=10, scale=1, num_channels=1, percentage = 1, )
nm_list = []
for name in test_dataset.key_list:
    
    nm_list.append(int(name.split("_")[3][:-3]))
# for each pkl file check if available


for i in range(len(results)):
    my_dict = results.iloc[i].to_dict()


    del my_dict['index']

    # check if model available
    if not os.path.exists(oj(model_path,my_dict['file_name']+".pt")):
        print(f"Model {my_dict['file_name']} not available")
        continue
    if "split_accs" in my_dict and type(my_dict['split_accs']) == dict:
        print(f"Model {my_dict['file_name']} already has split accs")
        continue
    # load model
    try:
        my_model =  UNet(n_channels = 1, 
                    n_classes =1, 
                    bilinear= True, 
                    in_factor = int(my_dict['in_factor']),
                    use_small = my_dict['reduce_layer'] == 1)

        my_model.load_state_dict(torch.load(oj(model_path,my_dict['file_name']+".pt")),)

        my_model = my_model.to(device)
        my_model.eval();
    except:
        print(f"Model {my_dict['file_name']} could not be loaded")
        continue

    # load test data
    start_offset= my_dict['start_offset']
    pred_offset = my_dict['prediction_offset']
    test_dataset = H5pyDataset( oj(data_folder, "phase_field_test.h5"), offset=pred_offset, start_offset=start_offset, scale=1, num_channels=1, percentage = 1, )
    split_arr = data_utils.test_trajectory(
        my_model,
        test_dataset,
        start_offset,
        40000,
        pred_offset,
        device=device,
    )[2]
    # print(split_arr)
    split_dict = {}
    for nm_val, err_val in zip(nm_list, split_arr):
        if nm_val not in split_dict and not np.isnan(err_val):
            split_dict[nm_val] = [err_val,]
        else:
            split_dict[nm_val].append(err_val)
    # probs not the best way but I am tired and coffe is not working
    split_dict = {key: np.mean(val) for key, val in split_dict.items()}
    my_dict['split_accs'] = split_dict
    pkl.dump(my_dict, open(oj(model_path,my_dict['file_name']+".pkl"), "wb"))



