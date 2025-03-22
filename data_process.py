
from torch.utils.data import DataLoader,Dataset,IterableDataset
import torch.utils.data.datapipes as dp
import torch
import xarray as xr
#to compare with billy
from datetime import datetime, timezone
import numpy as np

class ClimaticData(Dataset):
    def __init__(self,dataset_path, nsteps=1,tune_step=None,normalize=True):
        with xr.open_mfdataset(paths = dataset_path, engine='netcdf4',coords='minimal',compat='override',data_vars='all').to_array() as dataset:
            self.data = dataset.load()
        self.dt64 = self.data.time
        self.nsteps = nsteps
        self.normalize = normalize
        self.tune_step = tune_step

        if self.normalize:
            self.data_mean = torch.as_tensor(self.data.mean(dim='time').to_numpy(),dtype=torch.float32)
            self.data_std = torch.as_tensor(self.data.std(dim='time').to_numpy(),dtype=torch.float32)
        
    def __len__(self):
        #only cares about time series length
        if self.tune_step is None:
            factor = 1
        else:
            factor = self.tune_step

        return len(self.data['time']) - self.nsteps*factor
    
    def __getitem__(self,idx):
        inp = torch.as_tensor(self.data[:,idx,:,:].to_numpy(),dtype=torch.float32)
        if self.tune_step is None:
            tar = torch.as_tensor(self.data[:,idx+self.nsteps,:,:].to_numpy(),dtype=torch.float32)
        else:
            tar = torch.as_tensor(self.data[:,idx:idx+(self.nsteps*self.tune_step),:,:].to_numpy(),dtype=torch.float32) #returns [C,n_steps*tune_step,lat,long]
        #if not self.normalize:
        if self.normalize:
            if self.tune_step is None:
                inp=(inp - self.data_mean)/self.data_std
                #inp/=(self.data_std)

                tar=(tar - self.data_mean)/self.data_std #always broadcasteable
                #tar.div_(self.data_std)
            else:
                inp=(inp - self.data_mean)/self.data_std
                #inp/=(self.data_std)

                tar=(tar - self.data_mean.unsqueeze(1))/(self.data_std.unsqueeze(1)) #always broadcasteable
                #tar.div_(self.data_std)

        return inp.clone(),tar.clone()
    
    def _dt64_to_dt(self, dt64):
        ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    
    def get_timestamps_as_floats(self):
        timestamps = np.array([(dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's') for dt64 in self.dt64])
        return timestamps
    def get_timestamps_as_dates(self):
        return np.array(self.dt64,dtype=np.datetime64)
    def set_mean(self,mean_value):
        self.data_mean = mean_value
        return None
    
    def set_var(self,var_value):
        self.data_std = var_value
        return None
    
    def get_mean(self):
        return self.data_mean  
    def get_var(self):
        return self.data_std
    def get_lat(self):
        return len(self.data.lat)
    def get_lon(self):
        return len(self.data.lon)
    def get_channels(self):
        return len(self.data.variable)
    def data_keys(self):
        return list(self.data.coords["variable"].values)
    def get_temp_res(self):
        return self.nsteps

class MyIterableDataset(IterableDataset):
     def __init__(self, path):
         super(MyIterableDataset).__init__()
         # this depends on your dataset, suppose your dataset contains 
         # images whose path you save in this list
         self.dataset = path

     def __iter__(self):
         for image_path in self.dataset:
             sample, label = 'teste'# read an individual sample and its label
             yield sample, label


def format_outputs(file_path):
    data = torch.load(file_path, weights_only=False)

    time_stamps_all = list(data['predictions'].keys())[:1448]

    time_stamps_all = np.array(time_stamps_all)
    dict_fin = {'ground_truth':0,'predictions':0}
    dict_gt={}
    dict_pred={}

    for i in range(0,1448,8):
        dict_gt.update({time_stamps_all[i]:{ts:data['ground_truth'][ts] for ts in time_stamps_all[i:i+8]}})
        dict_pred.update({time_stamps_all[i]:{ts:data['predictions'][ts] for ts in time_stamps_all[i:i+8]}})

    dict_fin['predictions'] = dict_pred
    dict_fin['ground_truth'] = dict_gt 

    torch.save(dict_fin,file_path)