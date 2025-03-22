
#from torch_harmonics.examples.sfno import SphericalFourierNeuralOperatorNet as SFNO

#from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperatorBlock as SFBlock;

from torch_harmonics.sht import RealSHT, InverseRealSHT

import torch

import torch.nn as nn

from functools import partial

import math

from layers_utils import DropPath,MLP,SpectralConvS2


class SFBlock(nn.Module):
    """
    Helper module for a single SFNO/FNO block. Can use both FFTs and SHTs to represent either FNO or SFNO blocks.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        input_dim,
        output_dim,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.Identity,
        inner_skip="none",
        outer_skip="identity",
        use_mlp=True,
    ):
        super().__init__()

        if act_layer == nn.Identity:
            gain_factor = 1.0
        else:
            gain_factor = 2.0

        if inner_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.0

        self.global_conv = SpectralConvS2(forward_transform, inverse_transform, input_dim, output_dim, gain=gain_factor, bias=False)

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(input_dim, output_dim, 1, 1)
            nn.init.normal_(self.inner_skip.weight, std=math.sqrt(gain_factor / input_dim))
        elif inner_skip == "identity":
            assert input_dim == output_dim
            self.inner_skip = nn.Identity()
        elif inner_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {inner_skip}")

        # first normalisation layer
        self.norm0 = norm_layer()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        gain_factor = 1.0
        if outer_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.0

        if use_mlp == True:
            mlp_hidden_dim = int(output_dim * mlp_ratio)
            self.mlp = MLP(
                in_features=output_dim, out_features=input_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_rate=drop_rate, checkpointing=False, gain=gain_factor
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(input_dim, input_dim, 1, 1)
            torch.nn.init.normal_(self.outer_skip.weight, std=math.sqrt(gain_factor / input_dim))
        elif outer_skip == "identity":
            assert input_dim == output_dim
            self.outer_skip = nn.Identity()
        elif outer_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {outer_skip}")

        self.act_layer0 = act_layer()
        # second normalisation layer
        self.norm1 = norm_layer()

    def forward(self, x):

        x, residual = self.global_conv(x)

        x = self.norm0(x)

        if hasattr(self, "inner_skip"):
            x = x + self.inner_skip(residual)

        if hasattr(self,'act_layer0'):
            x = self.act_layer0(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.norm1(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            x = x + self.outer_skip(residual)

        return x



class SFNO(nn.Module):
   
    def __init__(
        self,
        img_size=(128, 256),
        grid="equiangular",
        grid_internal="legendre-gauss",
        scale_factor=3,
        in_chans=3,
        out_chans=3,
        embed_dim=256,
        num_layers=4,
        activation_function="gelu",
        encoder_layers=1,
        use_mlp=True,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        normalization_layer="none",
        hard_thresholding_fraction=1.0,
        use_complex_kernels=True,
        big_skip=False,
        pos_embed=False,
        inner_skip="none",
        outer_skip="identity",
    ):

        super().__init__()

        self.img_size = img_size
        self.grid = grid
        self.grid_internal = grid_internal
        self.scale_factor = scale_factor
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.normalization_layer = normalization_layer
        self.use_mlp = use_mlp
        self.encoder_layers = encoder_layers
        self.big_skip = big_skip

        # activation function
        if activation_function == "relu":
            self.activation_function = nn.ReLU
        elif activation_function == "gelu":
            self.activation_function = nn.GELU
        # for debugging purposes
        elif activation_function == "identity":
            self.activation_function = nn.Identity
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # compute downsampled image size. We assume that the latitude-grid includes both poles
        self.h = (self.img_size[0] - 1) // scale_factor + 1
        self.w = self.img_size[1] // scale_factor

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            norm_layer0 = partial(nn.LayerNorm, normalized_shape=(self.img_size[0], self.img_size[1]), eps=1e-6)
            norm_layer1 = partial(nn.LayerNorm, normalized_shape=(self.h, self.w), eps=1e-6)
        elif self.normalization_layer == "instance_norm":
            norm_layer0 = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
            norm_layer1 = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        elif self.normalization_layer == "none":
            norm_layer0 = nn.Identity
            norm_layer1 = norm_layer0
        else:
            raise NotImplementedError(f"Error, normalization {self.normalization_layer} not implemented.")

        if pos_embed == "latlon" or pos_embed == True:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], self.img_size[1]))
            nn.init.constant_(self.pos_embed, 0.0)
        elif pos_embed == "lat":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], 1))
            nn.init.constant_(self.pos_embed, 0.0)
        elif pos_embed == "const":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
            nn.init.constant_(self.pos_embed, 0.0)
        else:
            self.pos_embed = None

        # construct an encoder with num_encoder_layers
        num_encoder_layers = 1
        encoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        current_dim = self.in_chans
        encoder_layers = []
        for l in range(num_encoder_layers - 1):
            fc = nn.Conv2d(current_dim, encoder_hidden_dim, 1, bias=True)
            # initialize the weights correctly
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            encoder_layers.append(fc)
            encoder_layers.append(self.activation_function())
            current_dim = encoder_hidden_dim
        fc = nn.Conv2d(current_dim, self.embed_dim, 1, bias=False)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        encoder_layers.append(fc)
        self.encoder = nn.Sequential(*encoder_layers)

        # compute the modes for the sht
        modes_lat = self.h
        # due to some spectral artifacts with cufft, we substract one mode here
        modes_lon = (self.w // 2 + 1) - 1

        modes_lat = modes_lon = int(min(modes_lat, modes_lon) * self.hard_thresholding_fraction)

        self.trans_down = RealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid=self.grid).float()
        self.itrans_up = InverseRealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid=self.grid).float()
        self.trans = RealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=grid_internal).float()
        self.itrans = InverseRealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=grid_internal).float()

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):

            first_layer = i == 0
            last_layer = i == self.num_layers - 1

            if first_layer:
                norm_layer = norm_layer1
            elif last_layer:
                norm_layer = norm_layer0
            else:
                norm_layer = norm_layer1

            block = SFBlock(
                self.trans_down if first_layer else self.trans,
                self.itrans_up if last_layer else self.itrans,
                self.embed_dim,
                self.embed_dim,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_layer=norm_layer,
                use_mlp=use_mlp,
            )

            self.blocks.append(block)

        # construct an decoder with num_decoder_layers
        num_decoder_layers = 1
        decoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        current_dim = self.embed_dim + self.big_skip * self.in_chans
        decoder_layers = []
        for l in range(num_decoder_layers - 1):
            fc = nn.Conv2d(current_dim, decoder_hidden_dim, 1, bias=True)
            # initialize the weights correctly
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            decoder_layers.append(fc)
            decoder_layers.append(self.activation_function())
            current_dim = decoder_hidden_dim
        fc = nn.Conv2d(current_dim, self.out_chans, 1, bias=False)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        decoder_layers.append(fc)
        self.decoder = nn.Sequential(*decoder_layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):

        if self.big_skip:
            residual = x

        x = self.encoder(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.forward_features(x)

        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        x = self.decoder(x)

        return x






class MyModel(SFNO):
    def __init__(self,
                 nlat,
                 nlon,
                 n_channels,
                 normalization_layer="none",
                 activation_function="gelu",
                 inner_skip="none",
                 outer_skip="identity",
                 num_layers = 8,
                 embed_dim = 384,
                 drop_rate = 0.0,
                 drop_path = 0.0):

        super(MyModel,self).__init__()

        print("model will be created with:")

        print(f"inner_skip: {inner_skip} | outer_skip: {outer_skip}")
        print(f"normalization_layer: {normalization_layer} | activation: {activation_function}")
        print(f"n_layers: {num_layers} | embed_dim: {embed_dim}")

        self.model = SFNO(
                    img_size=(nlat, nlon),
                    grid="equiangular",
                    in_chans=n_channels,
                    out_chans=n_channels,
                    num_layers=num_layers, 
                    scale_factor=3, 
                    embed_dim=embed_dim,
                    big_skip=True, 
                    inner_skip=inner_skip,
                    outer_skip=outer_skip,
                    pos_embed="lat", 
                    use_mlp=True, 
                    normalization_layer=normalization_layer,
                    activation_function=activation_function,
                    drop_path_rate=drop_path,drop_rate=drop_rate)

        self.current_valid_ave = float('inf')

        #self.init_weights()
    def forward(self,x):

        x= self.model.forward(x)

        return x
    def set_ave_valid(self,new_value):
        self.current_valid_ave = new_value

    def get_best_val_loss(self):
        return self.current_valid_ave
    
    def set_best_val_loss(self,value):
        self.current_valid_ave=value

    def init_weights(self):
        # for param in self.model.parameters():
        #     #if len(param.shape) > 1:  # Skip biases
        #     nn.init.xavier_normal_(param) 
        for m in self.model.modules():
            # print("layer: ",m)
            if hasattr(m, 'weight') and (m.weight is not None) and (m.weight.dim() >=2):  # Check if layer has weights
                nn.init.xavier_normal_(m.weight)
                   

    def show_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"Layer: {name} | Mean weight: {param.data.mean():.6f} | std weight: {param.data.std():.6f} |\n Mean gradient: {param.grad.mean()  if param.grad is not None else 'None'} | std gradient: {param.grad.std()  if param.grad is not None else 'None'} | grad norm: {param.grad.data.norm(2)}")
            if param.grad is None:
                print(f"{name} has NO gradient!")

    def save_model(self,
                   epoch,
                   optimizer,
                   scheduler,
                   loss,
                   path):
        torch.save({'epoch':epoch,
                    'model_state_dict':self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss':loss,
                    'best_valid_loss': self.current_valid_ave},path)
        return None
        
    def load_model(self,path):
        checkpoint = torch.load(path,weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_valid_ave = checkpoint['best_valid_loss']
        return checkpoint['epoch'],checkpoint['loss'],checkpoint['optimizer_state_dict'],checkpoint['scheduler_state_dict']


