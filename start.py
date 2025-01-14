import torch.utils.data.datapipes as dp
import xarray as xr
import numpy as np
import torch
import sys

import torch.nn as nn
from torch.utils.data import DataLoader

from data_process import ClimaticData
from model import MyModel
from train_func import train

####
#Initiating a model for the first time
#arg1: dataset directory
#arg2: checkpoint output directory

###
#Datasets' setup
folder = '/home/ensta/ensta-guimaraes/projet/era_5_data'
save_model_path = '/home/ensta/ensta-guimaraes/projet/checkpoint_model.pth.tar'
train_years = list(range(1979,2016))
valid_years = list(range(2016,2018))

train_years = tuple('_{}_5.625deg.nc'.format(item) for item in train_years)
valid_years = tuple('_{}_5.625deg.nc'.format(item) for item in valid_years)

datapipe_train = dp.iter.FileLister([folder],recursive=True).filter(filter_fn=lambda filename: filename.endswith(train_years))
datapipe_valid = dp.iter.FileLister([folder],recursive=True).filter(filter_fn=lambda filename: filename.endswith(valid_years))

train_dataset = ClimaticData(dataset_path=datapipe_train)
valid_dataset = ClimaticData(dataset_path=datapipe_valid)

valid_dataset.set_mean(train_dataset.get_mean())
valid_dataset.set_var(train_dataset.get_var())
###
#Training parameters
batch_size = 128

n_epochs = 40

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
###
#Dataloaders setup

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=4)

valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=4)

###
#Model creation and optimizer

lat = train_dataset.get_lat()
lon = train_dataset.get_lon()
n_channels = train_dataset.get_channels()

model_holder = MyModel(lat,lon,n_channels)

optimizer = torch.optim.Adam(model_holder.model.parameters(), lr=3E-3, weight_decay=0.0)
###
#Train
train(model_holder,train_dataloader,valid_dataloader,optimizer,device,num_epochs=n_epochs,current_epoch=0,save_path=save_model_path)

