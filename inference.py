import torch.utils.data.datapipes as dp
import xarray as xr
import numpy as np
import torch
import sys

import torch.nn as nn
from torch.utils.data import DataLoader

from data_process import ClimaticData
from model import MyModel
from train_func import train, evaluate_year,plot_data,format_outputs

import matplotlib.pyplot as plt

import os

####
#Initiating a model for the first time
#arg1: dataset directory
#arg2: checkpoint output directory

###
#Datasets' setup
root_folder = '/home/ensta/ensta-guimaraes/projet'
folder = os.path.join(root_folder,'era_5_data')
#folder = '/home/ensta/ensta-guimaraes/projet/era_5_data'
save_model_path = os.path.join(root_folder,'checkpoint_model.pth.tar')
#save_model_path = '/home/ensta/ensta-guimaraes/projet/checkpoint_model.pth.tar'
save_loss = os.path.join(root_folder,'loss_model.pt')
#save_loss = '/home/ensta/ensta-guimaraes/projet/loss_model.pt'
save_outputs = os.path.join(root_folder,'output_model.pt')
#save_outputs = '/home/ensta/ensta-guimaraes/projet/output_model.pt'

train_years = list(range(1979,2016))
test_years = list(range(2018,2019))

train_years = tuple('_{}_5.625deg.nc'.format(item) for item in train_years)
test_years = tuple('_{}_5.625deg.nc'.format(item) for item in test_years)

datapipe_train = dp.iter.FileLister([folder],recursive=True).filter(filter_fn=lambda filename: filename.endswith(train_years))
datapipe_test = dp.iter.FileLister([folder],recursive=True).filter(filter_fn=lambda filename: filename.endswith(test_years))

train_dataset = ClimaticData(dataset_path=datapipe_train)
test_dataset = ClimaticData(dataset_path=datapipe_test)

test_dataset.set_mean(train_dataset.get_mean())
test_dataset.set_var(train_dataset.get_var())
###
#Training parameters
batch_size = 128

n_epochs = 40

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
###
#Dataloaders setup

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=4)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,pin_memory=True,num_workers=4)

###
#Model loading

lat = train_dataset.get_lat()
lon = train_dataset.get_lon()
n_channels = train_dataset.get_channels()

model_holder = MyModel(lat,lon,n_channels)

_,_,_ = model_holder.load_model(save_model_path)

model_holder.model.to(device)

output_length = (len(test_dataloader) + 6)//6


outcomes_holder = torch.empty((output_length,n_channels,lat,lon),dtype=torch.float32,device=device)
loss_holder = torch.empty(output_length,dtype=torch.float32,device=device)

###
#Inference

loss, outputs = evaluate_year(model_holder,test_dataloader,outcomes_holder,loss_holder,device)

###
#De-normalize data

ave_data = train_dataset.get_mean().to(device)
var_data = train_dataset.get_var().to(device)

outputs *= var_data.unsqueeze(0)
outputs += ave_data.unsqueeze(0)

###
#Saves everything

torch.save(loss, save_loss)
torch.save(outputs, save_outputs)

###
#Plotting and save image

model_holder.model.to('cpu')

loss = loss.cpu()
outputs = outputs.cpu()

fig = plt.figure(figsize=(12,6))

proj, Lats, Lons = format_outputs(model_holder)

plot_data(fig,proj,Lats,Lons,outputs[1],title='One autogresseive step')

plt.draw()

plt.save(os.path.join(root_folder,'test_plot.png'))

plt.close()
