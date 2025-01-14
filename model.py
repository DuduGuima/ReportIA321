
from torch_harmonics.examples.sfno import SphericalFourierNeuralOperatorNet as SFNO
import torch


class MyModel(SFNO):
    def __init__(self,nlat,nlon,n_channels):

        super(MyModel,self).__init__()

        self.model = SFNO(spectral_transform='sht', operator_type='driscoll-healy', img_size=(nlat, nlon), grid="equiangular",
                 in_chans=n_channels,out_chans=n_channels,num_layers=8, scale_factor=3, embed_dim=384, big_skip=True, pos_embed="lat", use_mlp=False, normalization_layer="none")
        
        self.current_valid_ave = float('inf')
    def forward(self,x):

        x= self.model.forward(x)

        return x
    def set_ave_valid(self,new_value):
        self.current_valid_ave = new_value
    def save_model(self,epoch,optimizer,loss,path):
        torch.save({'epoch':epoch,'model_state_dict':self.model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss':loss,'best_valid_loss': self.current_valid_ave},path)
        return None
        
    def load_model(self,path):
        checkpoint = torch.load(path,weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_valid_ave = checkpoint['best_valid_loss']
        return checkpoint['epoch'],checkpoint['loss'],checkpoint['optimizer_state_dict']


