
from torch.utils.data import DataLoader,Dataset,IterableDataset
import torch.utils.data.datapipes as dp
import torch
import xarray as xr

class ClimaticData(Dataset):
    def __init__(self,dataset_path, nsteps=6,normalize=True):
        with xr.open_mfdataset(paths = dataset_path, engine='netcdf4',coords='minimal',compat='override',data_vars='all').to_array() as dataset:
            self.data = dataset.load()

        self.nsteps = nsteps
        self.normalize = normalize

        if self.normalize:
            self.data_mean = torch.as_tensor(self.data.mean(dim='time').to_numpy(),dtype=torch.float32)
            self.data_var = torch.as_tensor(self.data.var(dim='time').to_numpy(),dtype=torch.float32)
        
    def __len__(self):
        #only cares about time series length
        return len(self.data['time']) - self.nsteps
    
    def __getitem__(self,idx):
        inp = torch.as_tensor(self.data[:,idx,:,:].to_numpy(),dtype=torch.float32)
        tar = torch.as_tensor(self.data[:,idx+self.nsteps,:,:].to_numpy(),dtype=torch.float32)
        
        if not self.normalize:
            inp.sub_(self.data_mean)
            inp.div_(self.data_var)

            tar.sub_(self.data_mean)
            tar.div_(self.data_var)

        return inp.clone(),tar.clone()
    
    def set_mean(self,mean_value):
        self.data_mean = mean_value
        return None
    
    def set_var(self,var_value):
        self.data_var = var_value
        return None
    
    def get_mean(self):
        return self.data_mean  
    def get_var(self):
        return self.data_var
    def get_lat(self):
        return len(self.data.lat)
    def get_lon(self):
        return len(self.data.lon)
    def get_channels(self):
        return len(self.data.variable)

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